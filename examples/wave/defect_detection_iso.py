
import sys
import os
import wave 
import random
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm
from PIL import Image
import argparse
import torch
import torch.nn as nn 
import numpy as np
import re
import gmsh
import numpy as np
import torch 
import torch.nn as nn
import scipy.sparse
import hashlib
import inspect
import meshio
import warnings
from itertools import chain
from functools  import reduce
from operator import eq
from collections import defaultdict
from typing import Iterable

from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
import matplotlib.tri as tri
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import animation
from abc import abstractclassmethod, abstractmethod
from numpy.polynomial.legendre import leggauss

import torch
from torch.autograd import Function
import scipy.sparse

topological_dimension = {
    "vertex": 0,
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "triangle6": 2,
    "quad9": 2,
}



class SparseMMScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):
        edata, row, col, B = ctx.saved_tensors
        edata_grad  = (grad_outputs[row] * B[col]).sum(dim=1)
       
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B

class SparseMVScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):

        edata, row, col, B = ctx.saved_tensors  
        edata_grad  = grad_outputs[row] * B[col]
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B
    
def spmv(edata, row, col, shape, B):
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    assert B.dim() == 1
    return SparseMVScipy.apply(edata, row, col, shape, B)

def spmm(edata, row, col, shape, B):
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    if B.dim() == 1:
        return spmv(edata, row, col, shape, B)
    assert B.dim() == 2
    SparseMMScipy.apply(edata, row, col, shape, B)

class SparseSolveScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b):
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsr()
        u = scipy.sparse.linalg.spsolve(A_scipy, b)
        u = torch.tensor(u, dtype=b.dtype)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        A_T             = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=(ctx.A_shape[1], ctx.A_shape[0])).tocsr()
        b_grad          = scipy.sparse.linalg.spsolve(A_T, grad_output.numpy())
        b_grad          = torch.tensor(b_grad, dtype=u.dtype)

        edata_grad      = - b_grad[row] * u[col] 
        return edata_grad, None, None, None, b_grad

def spsolve(edata, row, col, shape, b, backend=None, verbose=True):
    assert edata.device == row.device == col.device == b.device, f"edata, row, col b should be on the same device, but got {edata.device}, {row.device}, {col.device}, {b.device}"
    assert backend  in [None, "scipy", "petsc", "cupy", "torch"], f"backend should be None, scipy, petsc or cupy, but got {backend}"
    if edata.dtype != torch.float64:
        warnings.warn("Accuracy insufficient, float64 is recommended for better accuracy in spsolve")
    assert len(b.shape) <= 2, f"b should be of shape [n_node] or [n_node,batch], but got {b.shape}"
    return SparseSolveScipy.apply(edata, row, col, shape, b)

class SparseMatrix(nn.Module):
    def __init__(self, edata,  row, col, shape):
        super().__init__()
        assert edata.shape[0] == row.shape[0] == col.shape[0], f"the first dim of edata, row, col should be the same, but got {edata.shape[0]}, {row.shape[0]}, {col.shape[0]}"
        assert edata.device == row.device == col.device, f"edata, row, col should be on the same device, but got {edata.device}, {row.device}, {col.device}"
        self.register_buffer("edata", edata)
        self.register_buffer("row", row.long())
        self.register_buffer("col", col.long())
      
        self.shape = shape

        self.layout_hash = hashlib.sha256(self.row.cpu().numpy().tobytes() + self.col.cpu().numpy().tobytes()).hexdigest()

    def elementwise_operation(self, func, obj):
        """Elementwise operation with another sparse matrix or a tensor or a scalar
        If the object is a sparse matrix, the :attr:`edges` of the two sparse matrices should be the same

        Parameters
        ----------

        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            the elementwise operation
        obj: SparseMatrix or torch.Tensor or int or float
            the object to be elementwise operated with

        Returns
        -------
        result: SparseMatrix
            the result of the elementwise operation
        """
        if  isinstance(obj, SparseMatrix):
            assert self.shape == obj.shape, f"the shape of the two sparse matrices should be the same, but got {self.shape}, {obj.shape}"
            assert self.has_same_layout(obj), f"the row indices of the two sparse matrices should be the same, but got {self.row}, {obj.row}"
            return SparseMatrix(func(self.edata, obj.edata), self.row, self.col, self.shape)
        elif isinstance(obj, torch.Tensor):
            assert obj.shape == self.shape, f"the shape of the sparse matrix and the tensor should be the same, but got {self.shape}, {obj.shape}"
            return SparseMatrix(func(self.edata, obj), self.row, self.col, self.shape)
        elif isinstance(obj, (int,float)):
            return SparseMatrix(func(self.edata, obj), self.row, self.col, self.shape)
        else:
            raise Exception(f"unsupported type {type(obj)} for SparseMatrix.elementwise_operation {inspect.getsource(func)}")

    def __add__(self, obj):
        return self.elementwise_operation(lambda x,y: x+y, obj)

    def __mul__(self, obj):
        return self.elementwise_operation(lambda x,y: x * y, obj)

    def __rmul__(self, obj):
        return self.elementwise_operation(lambda a,b : torch.mul(b, a), obj)

    def __div__(self, obj):
        return self.elementwise_operation(torch.div, obj)
    
    def __rtruediv__(self, obj):
        return self.elementwise_operation(lambda a,b : torch.div(b, a), obj)
        
    def __pow__(self, obj):
        return self.elementwise_operation(torch.pow, obj)

    def __matmul__(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            the dense tensor of shape [b] or [b,h] to be multiplied with the sparse matrix

        Returns
        -------
        torch.Tensor
            the result of the multiplication of shape [a] or [a,h]
        """
        assert x.shape[0] == self.shape[1], f"the first dim of x should be the same as the second dim of the sparse matrix, but got [{self.shape[0]},{self.shape[1]}] @ [{x.shape[0]},..] "
        assert x.device == self.edata.device, f"the device of x should be the same as the device of the sparse matrix, but got {x.device}, {self.edata.device}"
        return spmm(self.edata, self.row, self.col, self.shape, x)

    def __neg__(self):
        return SparseMatrix(-self.edata, self.row, self.col, self.shape)

    def solve(self, x, backend=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            the dense tensor of shape [a] or [a,h] to be solved with the sparse matrix
        backend: str, optional
            the backend to solve the sparse matrix, can be :obj:`None`, :obj:`"torch"`, :obj:`"scipy"` 
            or :obj:`"torch_scipy"`, default :obj:`None`
        
        Returns
        -------
        torch.Tensor
            the result of the solution of shape [b] or [b,h]
        """
        assert x.shape[0] == self.shape[1], f"the first dim of x should be the same as the second dim of the sparse matrix, but got {x.shape[0]}, {self.shape[1]}"
        assert x.device == self.edata.device, f"the device of x should be the same as the device of the sparse matrix, but got {x.device}, {self.edata.device}"
        return spsolve(self.edata, self.row, self.col, self.shape, x, backend=backend)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Parameters
        ----------
        requires_grad: bool, optional
            whether the sparse matrix requires gradient, default :obj:`True`
        
        Returns
        -------
        SparseMatrix    
            the sparse matrix with requires_grad set to requires_grad
        """
        self.edata.requires_grad_(requires_grad)
        return self
  
    def transpose(self):
        """tranpose the sparse matrix

        .. math::
            
            A = A^\\top
        
        """
        return SparseMatrix(self.edata, self.col, self.row, self.shape[::-1])

    def sqrt(self):
        """element-wise square root
        
        .. math::

            A_{ij} = \\sqrt{A_{ij}}
        
        """
        return SparseMatrix(self.edata.sqrt(), self.row, self.col, self.shape)
    
    def reciprocal(self):
        """element-wise reciprocal
        
        .. math::
            
            A_{ij} = \\frac{1}{A_{ij}}

        """
        return SparseMatrix(self.edata.reciprocal(), self.row, self.col, self.shape)

    def degree(self, axis=0):
        """how many non-zero element in each row/column
        - axis = :obj:`0`
            
            .. math::
                \\sum_{j}\mathbb{1}_{A_{ij} \\neq 0}    
        
        - axis = :obj:`1`
        
            .. math::
                \\sum_{i}\mathbb{1}_{A_{ij} \\neq 0}     
        
        Parameters
        ----------
        axis: int, optional
            the axis to sum, can be :obj:`0` or :obj:`1`, default :obj:`0`
        
        Returns
        -------
        torch.Tensor
            the degree of shape :math:`[n_{\\text{row}}]` or :math:`[n_{\\text{col}}]`
        """
        nonzero = self.row if axis == 0 else self.col
        return torch.bincount(nonzero, minlength=self.shape[0] if axis == 0 else self.shape[1])

    def sum(self, axis=None):
        """sum of all non-zero elements

        * axis = :obj:`None`

            .. math::
                \sum_{ij}A_{ij}

        * axis = :obj:`0`

            .. math::
                \sum_{j}A_{ij}
        
        * axis = :obj:`1`
        
            .. math::
                \sum_{i}A_{ij}

        Parameters
        ----------
        axis: int, optional
            the axis to sum, can be :obj:`None`, :obj:`0` or :obj:`1`, default :obj:`None`
        
        Returns
        -------
        torch.Tensor
            the sum of shape :math:`[]` or :math:`[n_\\text{row}]` or :math:`[n_\\text{col}]`
        """
        if axis is None:
            return self.edata.sum()
        elif axis == 0:
            return self.T @ torch.ones(self.shape[0], device=self.edata.device)
        elif axis == 1:
            return self @ torch.ones(self.shape[1], device=self.edata.device)
        else:
            raise Exception(f"unsupported axis {axis} for SparseMatrix.sum")

    def clone(self):
        """The cloned sparse matrix will share clone gradient with the original sparse matrix
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the cloned sparse matrix
        """
        return SparseMatrix(self.edata.clone(), self.row.clone(), self.col.clone(), self.shape)
    
    @property
    def T(self):
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the transpose of the sparse matrix
        """
        return self.transpose()
    
    @property
    def requires_grad(self):
        """
        Returns
        -------
        bool
            whether the sparse matrix requires gradient or not
        """
        return self.edata.requires_grad

    @property
    def dtype(self):
        """
        Returns
        -------
        torch.dtype
            the dtype of the sparse matrix
        """
        return self.edata.dtype
    
    @property
    def device(self):
        """
        Returns
        -------
        torch.device
            the device of the sparse matrix
        """
        return self.edata.device

    @property
    def grad(self):
        """
        Returns
        ------- 
        torch_fem.sparse.SparseMatrix or None
            if the sparse matrix requires gradient, return the grad for each element 
            of the sparse matrix, otherwise return :obj:`None`
        """
        if self.edata.grad is None:
            return None
        else:
            return SparseMatrix(self.edata.grad, self.row, self.col, self.shape)

    @property
    def grad_fn(self):
        """
        Returns
        -------
        torch.autograd.Function or None
            if the sparse matrix requires gradient, return the grad_fn for each element 
            of the sparse matrix, otherwise return :obj:`None`
        """
        if self.edata.grad_fn is None:
            return None
        else:
            return self.edata.grad_fn

    @property
    def nnz(self):
        """
        Returns
        -------
        int
            the number of non-zero elements
        """
        return self.edata.shape[0]

    @property
    def layout_mask(self):
        """
        Returns
        -------
        torch.Tensor
            the mask of the layout, where the non-zero elements are 1, otherwise 0
        """
        mask = torch.zeros(self.shape, device=self.edata.device)
        mask[self.row, self.col] = 1
        return mask

    def type(self, dtype):
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix with dtype set to dtype
        """
        self.edata.to_(dtype)
        return self

    def detach(self):
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix with requires_grad set to False
        """
        return SparseMatrix(self.edata.detach(), self.row, self.col, self.shape).requires_grad_(False)

    def to_scipy_coo(self):
        """
        Returns
        -------
        scipy.sparse.coo_matrix
            the scipy.sparse.coo_matrix of the sparse matrix
        """
        return scipy.sparse.coo_matrix((
            self.edata.detach().cpu().numpy(),
            (
                self.row.detach().cpu().numpy(),
                self.col.detach().cpu().numpy()
            )), shape=self.shape)
    
    def to_sparse_coo(self):
        """Turn the sparse matrix into a torch.sparse_coo_tensor, the gradient will be lost
        Returns
        -------
        torch.sparse_coo_tensor
            the torch.sparse_coo_tensor of the sparse matrix
        """
        return torch.sparse_coo_tensor(
            torch.stack([self.row, self.col]),
            self.edata,
            self.shape
        )

    def to_dense(self):
        """Turn the sparse matrix into a dense matrix, the gradient will be maintained
        Returns
        -------
        torch.Tensor
            the dense tensor of the sparse matrix
        """
        matrix = torch.zeros(self.shape, device=self.edata.device, dtype=self.edata.dtype)
        matrix[self.row, self.col] += self.edata
        return matrix

    def has_same_layout(self, obj):
        """
        Parameters
        ----------
        obj: SparseMatrix or str
            the object to be compared with, if it is a str, it will be compared with the layout_hash of the sparse matrix

        Returns
        -------
        bool
            whether the two sparse matrices have the same layout
        """
        assert isinstance(obj, (SparseMatrix,str)), f"matrix must be SparseMatrix or str, but got {type(obj)}"
        if  isinstance(obj, str):
            return self.layout_hash == obj
        else:
            return self.layout_hash == obj.layout_hash

tri_basis_p1 = torch.tensor([[0, 0],
                         [1, 0],
                         [0, 1]], dtype=torch.float)

def tri_shape_val_p1(quadrature):
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros((*quadrature.shape[:-1], 3), device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    phi[..., 0] = 1 - xi - eta
    phi[..., 1] = xi
    phi[..., 2] = eta
    return phi

def tri_shape_grad_p1(quadrature, element_coords, return_jac=False):
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 3, 2], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    grad_phi[..., 0, (0,1)] = -1
    grad_phi[..., 1, 0] = 1
    grad_phi[..., 2, 1] = 1
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi

tri_basis_p2 = torch.tensor([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=torch.float32)

def tri_shape_val_p2(quadrature):
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros(*quadrature.shape[:-1], 6, device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    
    phi[..., 0] = (1 - xi - eta) * ( 1 - 2*xi - 2*eta)
    phi[..., 1] = xi * (2*xi - 1)
    phi[..., 2] = eta * (2*eta - 1)
    phi[..., 3] = 4*xi * (1 - xi - eta)
    phi[..., 4] = 4*xi * eta
    phi[..., 5] = 4*eta * (1 - xi - eta)

    return phi

def tri_shape_grad_p2(quadrature, element_coords, return_jac=False):
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 6, 2], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_corner, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    n_basis = 6
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    grad_phi[..., 0, 0] = -3 + 4*xi + 4*eta
    grad_phi[..., 0, 1] = -3 + 4*xi + 4*eta
    grad_phi[..., 1, 0] = 2 - 4*xi - 2*eta
    grad_phi[..., 1, 1] = -2*xi
    grad_phi[..., 2, 0] = 2 - 4*xi - 2*eta
    grad_phi[..., 2, 1] = 0
    grad_phi[..., 3, 0] = 2*eta
    grad_phi[..., 3, 1] = 2*xi
    grad_phi[..., 4, 0] = -2*eta
    grad_phi[..., 4, 1] = 1 - 2*xi - 2*eta
    grad_phi[..., 5, 0] = 0
    grad_phi[..., 5, 1] = 4*eta - 2

    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi

quad_basis_p1 = torch.tensor([[0., 0.],
                        [1., 0.],
                        [1., 1.],
                        [0., 1.]], dtype=torch.float32) 

def quad_shape_val_p1(quadrature):
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    assert quadrature.dim() == 2, f"quadrature must be 2D, but got {quadrature.dim()}"

    phi = torch.zeros((*quadrature.shape[:-1], 4), device=quadrature.device,  dtype=quadrature.dtype)
    x, y = quadrature[..., 0], quadrature[..., 1]
    phi[:, 0] = (1. - x) * (1. - y)
    phi[:, 1] = x * (1. - y)
    phi[:, 2] = x * y
    phi[:, 3] = (1. - x) * y

    return phi

def quad_shape_grad_p1(quadrature, element_coords, return_jac=False):
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.shape[1:] == (4, 2), f"element_coords must be 3D of shape [n_element, 4, 2], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    x, y = quadrature[..., 0], quadrature[..., 1]
    grad_phi[:, 0, 0] = -1 + y
    grad_phi[:, 0, 1] = -1 + x 
    grad_phi[:, 1, 0] = 1 - y
    grad_phi[:, 1, 1] = -x
    grad_phi[:, 2, 0] = y
    grad_phi[:, 2, 1] = x
    grad_phi[:, 3, 0] = -y
    grad_phi[:, 3, 1] = 1 - x
    
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi

quad_basis_p2 = torch.tensor([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                        [0.5, 0.0],
                        [1.0, 0.5],
                        [0.5, 1.0],
                        [0.0, 0.5],
                        [0.5, 0.5]], dtype=torch.float32) 

def quad_shape_val_p2(quadrature):
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros(*quadrature.shape[:-1], 9, device=quadrature.device, dtype=quadrature.dtype)
    quadrature = 2 * quadrature - 1
    x, y = quadrature[..., 0], quadrature[..., 1]
    x2, y2 = x ** 2, y ** 2

    phi[:, 0] = 0.25 * (x2 - x) * (y2 - y)
    phi[:, 1] = 0.25 * (x2 + x) * (y2 - y)
    phi[:, 2] = 0.25 * (x2 + x) * (y2 + y)
    phi[:, 3] = 0.25 * (x2 - x) * (y2 + y)
    phi[:, 4] = 0.5 * (y2 - y) * (1 - x2)
    phi[:, 5] = 0.5 * (x2 + x) * (1 - y2)
    phi[:, 6] = 0.5 * (y2 + y) * (1 - x2)
    phi[:, 7] = 0.5 * (x2 - x) * (1 - y2)
    phi[:, 8] = (1 - x2) * (1 - y2)

    return phi

def quad_shape_grad_p2(quadrature, element_coords, return_jac=False):
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.shape[1:] == (9,2), f"element_coords must be 3D of shape [n_element, 9, 2], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_corner, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    n_basis = 9
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    quadrature = 2 * quadrature - 1
    x, y = quadrature[..., 0], quadrature[..., 1]
    x2, y2 = x ** 2, y ** 2
    grad_phi[:, 0, 0] = 0.25 * ((-1 + 2 * x) * (-1 + y) * y)
    grad_phi[:, 0, 1] = 0.25 * ((-1 + x) * x * (-1 + 2 * y))
    grad_phi[:, 1, 0] = 0.25 * ((1 + 2 * x) * (-1 + y) * y)
    grad_phi[:, 1, 1] = 0.25 * (x * (1 + x) * (-1 + 2 * y))
    grad_phi[:, 2, 0] = 0.25 * ((1 + 2 * x) * y * (1 + y))
    grad_phi[:, 2, 1] = 0.25 * (x * (1 + x) * (1 + 2 * y))
    grad_phi[:, 3, 0] = 0.25 * ((-1 + 2 * x) * y * (1 + y))
    grad_phi[:, 3, 1] = 0.25 * ((-1 + x) * x * (1 + 2 * y))
    grad_phi[:, 4, 0] = -(x * (-1 + y) * y)
    grad_phi[:, 4, 1] = -0.5 * ((-1 + x2) * (-1 + 2 * y))
    grad_phi[:, 5, 0] = -0.5 * ((1 + 2 * x) * (-1 + y2))
    grad_phi[:, 5, 1] = -(x * (1 + x) * y)
    grad_phi[:, 6, 0] = -(x * y * (1 + y))
    grad_phi[:, 6, 1] = -0.5 * ((-1 + x2) * (1 + 2 * y))
    grad_phi[:, 7, 0] = -0.5 * ((-1 + 2 * x) * (-1 + y2))
    grad_phi[:, 7, 1] = -((-1 + x) * x * y)
    grad_phi[:, 8, 0] = 2 * x * (-1 + y2)
    grad_phi[:, 8, 1] = 2 * (-1 + x2) * y

    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi


def get_quadrature(element_type, order:int=None):
    """get the quadrature information
    Parameters
    ----------
    element_type : str
        the type of the element
    order : int, optional
        the order of the quadrature, by default None
        
    Returns
    -------
    weights : torch.Tensor
        the weights of the quadrature
    points : torch.Tensor
        the points of the quadrature
    """
    if element_type.startswith("line"): # line
        if order is None:
            order = 1
        points, weights = leggauss(order+1)
        points = torch.from_numpy(0.5 * points + 0.5)[:, None]
        weights = torch.from_numpy(0.5 * weights)
    elif element_type.startswith("tri"): # triangle
        if order is None:
            order = 1
        order = str(order)
        assert order in quadrature_lookup["tri"], f"order must be one of {list(quadrature_lookup['tri'].keys())}, but got {order}"
        points = torch.tensor(quadrature_lookup["tri"][order]["points"])
        weights = torch.tensor(quadrature_lookup["tri"][order]["weights"])
    elif element_type.startswith("quad"): # quadrilateral
        if order is None:
            order = 2
        points, weights = leggauss(order+1)
        points = 0.5 * points + 0.5
        weights = 0.5 * weights
        points  = np.stack(np.meshgrid(points, points), -1).reshape(-1, 2)
        weights = np.outer(weights, weights).reshape(-1)
        points = torch.from_numpy(points)
        weights = torch.from_numpy(weights)
    elif element_type.startswith("tet"): # tetrahedron
        if order is None:
            order = 1
        order = str(order)
        assert order in quadrature_lookup["tetra"], f"order must be one of {list(quadrature_lookup['tetra'].keys())}, but got {order}"
        points = torch.tensor(quadrature_lookup["tetra"][order]["points"])
        weights = torch.tensor(quadrature_lookup["tetra"][order]["weights"])
    elif element_type.startswith("hex"): # hexahedron
        if order is None:
            order = 2
        points, weights = leggauss(int(np.ceil((order + 1.0) / 2.0)))
        points  = np.stack(np.meshgrid(points, points, points), -1).reshape(-1, 3)
        w1, w2, w3 = np.meshgrid(weights, weights, weights)
        weights = (w1 * w2 * w3).reshape(-1)
        points = torch.from_numpy(points)
        weights = torch.from_numpy(weights)
    else:
        raise ValueError(f"Unknown element type: {element_type}")
    return weights, points


def get_shape_val(element_type, quadrature_points):
    find_shape_val = {
        "triangle":tri_shape_val_p1,
        "triangle6":tri_shape_val_p2,
        "tri3":tri_shape_val_p1,
        "tri6":tri_shape_val_p2,
        "quad":quad_shape_val_p1,
        "quad4":quad_shape_val_p1,
        "quad9":quad_shape_val_p2,
    }
    if element_type not in find_shape_val:
        raise ValueError(f"element_type must be one of {list(find_shape_val.keys())}, but got {element_type}")

    return  find_shape_val[element_type](quadrature_points)
   
def get_shape_grad(element_type, quadrature_weights, quadrature_points, element_coords):
    find_shape_grad = {
        "triangle":tri_shape_grad_p1,
        "triangle6":tri_shape_grad_p2, 
        "tri3":tri_shape_grad_p1,
        "tri6":tri_shape_grad_p2,
        "quad":quad_shape_grad_p1,
        "quad4":quad_shape_grad_p1,
        "quad9":quad_shape_grad_p2,
    }

    assert element_type in find_shape_grad, f"element_type must be one of {list(find_shape_grad.keys())}, but got {element_type}"

    shape_grad, jac = find_shape_grad[element_type](quadrature_points, element_coords, return_jac=True) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, n_dim, n_dim]

    jacdet = torch.abs(torch.det(jac))
    jxw    = jacdet * quadrature_weights
    return shape_grad, jxw

class BufferDict(nn.Module):
    def __init__(self, data = None):
        super().__init__()
        if data is None:
            data = {}
        self._data = {} # used for storing data that cannot be used as a valid name
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
      
        for key in list(data.keys()):
            if not pattern.match(key):
                self._data[key] = data.pop(key)
        for key, value in data.items():
            if isinstance(value, nn.Module):
                setattr(self, key, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")
    
    def as_parameter(self, key):
        buffer = self._buffers.pop(key)
        self.register_parameter(key, buffer)
        
    def as_buffer(self, key):
        parameter = self._parameters.pop(key)
        self.register_buffer(key, parameter)
        
    def keys(self):
        return chain(self._buffers.keys(), self._parameters.keys(), self._data.keys(), self._modules.keys())
    
    def items(self):
        return chain(self._buffers.items(), self._parameters.items(), self._data.items(), self._modules.items())
    
    def values(self):
        return chain(self._buffers.values(), self._parameters.values(), self._data.values(), self._modules.values())
    
    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(f"{key} is not found in the BufferDict")
        
        return self._buffers[key] if key in self._buffers else self._parameters[key] if key in self._parameters else self._data[key] if key in self._data else self._modules[key]

    def __setitem__(self, key, value):
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        if not pattern.match(key):
            self._data[key] = value
        else:
            if isinstance(value, nn.Module):
                setattr(self, key, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")

    def __len__(self):
        return len(self._buffers) + len(self._parameters) + len(self._data)
    
    def __includes__(self, key):
        return key in self.keys()

    def is_floating_point(self):
        return any(map(lambda x:x.is_floating_point(), self.values()))

    def is_complex(self):
        return any(map(lambda x:x.is_complex(), self.values()))

    @property
    def dtype(self):
        return next(iter(self.buffers().values())).dtype

    @property
    def device(self):
        return next(iter(self.buffers().values())).device
    
    def _apply(self, fn):
        self = super()._apply(fn)
        for key, value in self._data.items():
            self._data[key] = fn(value)
        return self

    def __str__(self):
        return f"""BufferDict(
        {', '.join([f"{key} = {value}" for key, value in self.items()])}
        )"""
      
    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {key:value for key, value in self.items()}

    def clone(self):
        data = {key:value.clone() for key, value in self.items()}
        return BufferDict(data)

class Projector(nn.Module):
    def __init__(self, from_, to_, from_shape, to_shape, dtype = None):
        super().__init__()
        if isinstance(from_shape, int):
            from_shape = (from_shape,)
        elif isinstance(from_shape, np.ndarray):
            assert from_shape.ndim == 1, f"from_shape must be 1D, but got {from_shape.ndim}"

        if isinstance(to_shape, int):
            to_shape = (to_,)
        elif isinstance(to_shape, np.ndarray):
            assert to_shape.ndim == 1, f"to_shape must be 1D, but got {to_shape.ndim}"

        if isinstance(from_, np.ndarray):
            from_ = torch.from_numpy(from_)
        if isinstance(to_, np.ndarray):
            to_   = torch.from_numpy(to_)
        assert from_.shape == to_.shape, f"from_ and to_ must have the same shape, but got {from_.shape}, {to_.shape}"
        assert len(from_.shape) == 1, f"from_ and to_ must be 1D, but got {from_.shape}, {to_.shape}"
        if dtype is None:
            dtype = from_.dtype
        projection = torch.sparse_coo_tensor(
            torch.stack([to_,from_],0), 
            torch.ones_like(from_,dtype=torch.float32),
            size = (np.prod(to_shape), np.prod(from_shape))
        ).to_sparse_csr()
        self.register_buffer("projection", projection)
        self.from_shape = from_shape
        self.to_shape   = to_shape

    def type(self, dtype):
        if dtype != self.dtype:
            self.projection = self.projection.type(dtype)
        return self

    @property
    def device(self):
        return self.projection.device

    @property
    def dtype(self):
        return self.projection.dtype

    def __call__(self, x):
        assert self.dtype == x.dtype, f"the dtype of x must be {self.dtype}, but got {x.dtype}"
        assert self.device == x.device, f"the device of x must be {self.device}, but got {x.device}"
        assert x.shape[:len(self.from_shape)] == self.from_shape, f"the shape of x must be [{self.from_shape}, ...], but got {x.shape}"

        dim_shape = x.shape[len(self.from_shape):]
        x = x.reshape(np.prod(self.from_shape), -1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            x = (self.projection @ x).squeeze(-1)
        else:
            x = self.projection @ x
        x = x.reshape(*self.to_shape, *dim_shape)
        return x

class ElementAssembler(nn.Module):

    def __init__(self, quadrature_weights,
                        quadrature_points,
                        shape_val,
                        projector, 
                        elements,
                        edges,
                        n_points):
        super().__init__()
        element_types = list(quadrature_weights.keys())
        dimension     = 3

        self.quadrature_weights = quadrature_weights
        self.quadrature_points  = quadrature_points
        self.shape_val          = shape_val
        self.projector          = projector
        self.elements           = elements
        self.register_buffer("edges", edges)
        self.dimension          = dimension
        self.element_types      = list(elements.keys())
        self.n_points           = n_points
        self.__post_init__()

    def __post_init__(self):
        pass
    @property
    def device(self):
        return self.quadrature_weights.device

    @property
    def dtype(self):
        return self.quadrature_weights.dtype

    def type(self,  dtype):
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self

    def _integrate(self, batch_integral, jxw, n_element, n_basis, use_element_parallel):
        if use_element_parallel:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [{n_element}, batch_size, {n_basis},{n_basis}] or [{n_element}, batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 4 or batch_integral.dim() == 6, error_msg
            assert batch_integral.shape[0] == n_element, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg 
            assert batch_integral.shape[3] == n_basis, error_msg
            if batch_integral.dim() == 6:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("eqij...,eq->eij...", batch_integral, jxw)
        else:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [batch_size, {n_basis},{n_basis}] or [batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 3 or batch_integral.dim() == 5, error_msg
            assert batch_integral.shape[1] == n_basis, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg
            if batch_integral.dim() == 5:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("qij...,eq->eij...", batch_integral, jxw)

        return batch_integral
    
    def _build_output(self, integral):
        if integral.dim() == 1:
            return SparseMatrix(integral, self.edges[0], self.edges[1], shape=(self.n_points, self.n_points))
        else:
            raise Exception(f"the shape of integral is supposed to be  1D or 3D, but got {integral.shape}")

    def __call__(self, points, func=None,point_data=None, batch_size=1):
        assert isinstance(point_data, dict) or point_data is None, f"point_data should be a dict, but got {type(point_data)}. Please pass  in extra parameter using key-value pairs"
        if point_data is None:
            point_data = {}
        point_data["x"] = points

        self = self.type(points.dtype).to(points.device)

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], f"the shape of {key} should be [n_point, ...], but got {value.shape}"
 

        signature = inspect.signature(self.forward)

        fn = None
        
        use_element_parallel = None

        integral = None
      
        for element_type in self.element_types:
            element_integral = None
            n_quadrature = self.quadrature_weights[element_type].shape[0]
            n_batch      = n_quadrature // batch_size if batch_size is not None else 1
            n_batch_size = batch_size if batch_size is not None else n_quadrature
            n_basis      = self.shape_val[element_type].shape[1]
            n_element    = self.elements[element_type].shape[0]
            ele_point_data = {k:v[self.elements[element_type]] for k,v in point_data.items()}
            ele_coords   = points[self.elements[element_type]] # [n_element, n_basis, n_dim]
            for i in range(n_batch):
                shape_val = self.shape_val[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size, n_basis]
                w         = self.quadrature_weights[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size]
                quadrature_points = self.quadrature_points[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size, n_dim]
                shape_grad, jxw = get_shape_grad(element_type, w, quadrature_points, ele_coords) # [n_element, batch_size, n_basis, n_dim], [n_element, n_batch]
                
                # prepare arguments
                args = []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        args.append(shape_val)
                    elif key in ["gradu", "gradv"]:
                        args.append(shape_grad)
                    elif key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eq...",ele_point_data[key], shape_val))
                    elif key.startswith("grad") and key[4:] in ele_point_data:
                        args.append(torch.einsum("eb...,eqbd->eq...d",ele_point_data[key[4:]], shape_grad))
                    else:
                        raise NotImplementedError(f"key {key} is not implemented")



                # parallel dispatch 

                if fn is None:
                    element_dims = []
                    quadrature_dims = []
                    for key in signature.parameters:
                        if key in ["u", "v"]:
                            element_dims.append(None)
                            quadrature_dims.append(0)
                        else:
                            element_dims.append(0)
                            quadrature_dims.append(0)
                    
                    element_dims = tuple(element_dims)
                    quadrature_dims = tuple(quadrature_dims)

                    fn = self.forward if func is None else func
                   
                    if all([x is None for x in element_dims]):
                        # if all is shape_val
                        fn = torch.vmap(fn, in_dims=quadrature_dims)
                        use_element_parallel = False
                    else:
                        fn = torch.vmap(
                            torch.vmap(
                                fn,
                                in_dims = quadrature_dims
                            ),
                            in_dims=element_dims
                        )
                        use_element_parallel = True

                batch_integral = fn(*args) # [n_element, batch_size, n_basis, n_basis, ...] or [n_batch, batch_size, n_basis, ...]

                batch_integral = self._integrate(batch_integral, jxw, n_element, n_basis, use_element_parallel)

                if element_integral is None:
                    element_integral = batch_integral
                else:
                    element_integral += batch_integral
    
            if integral is None:
                integral = self.projector[element_type](element_integral) # [n_edge, ...]
            else:
                integral += self.projector[element_type](element_integral) # [n_edge, ...]

        return self._build_output(integral)

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError(f"forward is not implemented")
        


    @classmethod
    def from_assembler(cls, obj):
        assert isinstance(obj, ElementAssembler), f"obj must be an instance of ElementAssembler, but got {type(obj)}"
        return cls(obj.quadrature_weights,
                   obj.quadrature_points,
                   obj.shape_val,
                   obj.projector, 
                   obj.elements,
                   obj.edges,
                   obj.n_points)

    @classmethod
    def from_mesh(cls, mesh, quadrature_order=None):
        elements = mesh.elements()
        n_points = mesh.points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}

        quadrature_weights = {}
        quadrature_points  = {}
        shape_val          = {}
        projector          = {}
        
        elem_u, elem_v = [], []
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            quadrature_weights[element_type], quadrature_points[element_type] =\
            get_quadrature(element_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
            shape_val[element_type] = get_shape_val(element_type, quadrature_points[element_type]) # [n_quadrature, n_basis]
           
            for i in range(n_basis):
                for j in range(n_basis):
                    elem_u.append(value[:, i])
                    elem_v.append(value[:, j])

        elem_u, elem_v = torch.stack(elem_u, -1).flatten(), torch.stack(elem_v, -1).flatten() # [num_elements * num_basis * num_basis]
        elem_u, elem_v = elem_u.cpu().numpy().copy(), elem_v.cpu().numpy().copy()
        tmp = scipy.sparse.coo_matrix(( # used to remove duplicated edges
            np.ones_like(elem_u), # data
            (elem_u, elem_v), # (row, col)
        ), shape = (n_points,  n_points)).tocsr().tocoo()
        edge_u, edge_v = tmp.row, tmp.col
        num_edges  = len(edge_u)
        eids_csr = scipy.sparse.coo_matrix((
            np.arange(num_edges), (edge_u, edge_v)
        ), shape=(n_points, n_points)).tocsr()    
        elem_eids     = np.array(eids_csr[elem_u, elem_v].copy()).ravel()

        ptr = 0
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            elem_eids  = np.array(eids_csr[elem_u[ptr:ptr+n_element*n_basis*n_basis], elem_v[ptr:ptr+n_element*n_basis*n_basis]].copy()).ravel()
            ptr += n_element * n_basis * n_basis
            projector[element_type] = Projector(
                from_ = torch.arange(n_element * n_basis * n_basis), 
                to_    = torch.from_numpy(elem_eids),
                from_shape = (n_element, n_basis, n_basis), 
                to_shape = (num_edges,),
            )
         
        edges = torch.from_numpy(np.stack([edge_u, edge_v], 0))

        quadrature_weights = BufferDict(quadrature_weights)
        quadrature_points  = BufferDict(quadrature_points)
        shape_val          = BufferDict(shape_val)
        projector          = BufferDict(projector)
        elements           = BufferDict({k:v.long() for k,v in elements.items()})
        

        assembler = cls(quadrature_weights,
                   quadrature_points,
                   shape_val,
                   projector, 
                   elements,
                   edges,
                   n_points)
        assembler = assembler.type(mesh.dtype).to(mesh.device)
        return assembler

def draw_mesh(points, elements, value, ax=None, show_colorbar=True,show_mesh=False):
    assert points.shape[1] == 2, f"points must be 2D, but got {points.shape}"

    if ax is None:
        ax = plt.gca()

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(elements, torch.Tensor):
        elements = elements.detach().cpu().numpy()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if elements.shape[1] == 3: # tri
        triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
        img = ax.tripcolor(triang, value, cmap=plt.cm.jet, shading='gouraud')
        if show_mesh:
            ax.triplot(triang, color='k', linewidth=0.5)
    elif elements.shape[1] == 4: # quad 
        triang = tri.Triangulation(points[:, 0], points[:, 1], np.concatenate([elements[:,(0,1,2)],elements[:,(0,2,3)]],0))
        img = ax.tripcolor(triang, value, cmap=plt.cm.jet, shading='gouraud')
        if show_mesh:
            polygons = [patches.Polygon(points[element], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
    else:
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        z_grid = griddata(points, value, (x_grid, y_grid), method='linear')
        img = ax.imshow(z_grid.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=plt.cm.jet, aspect='auto')


        if show_mesh:
            polygons = [patches.Polygon(points[element], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)


    ax.axis("equal")
    ax.axis("off")

    if show_colorbar:
        cb = plt.colorbar(img, ax=ax)
        return img, cb
    
    return img

class StreamPlotter:
    def __init__(self,   nrows=1, ncols=1, width=5, height=5, filename=None):
        if filename is None:
            filename = "stream_plotter.mp4"
        fig, axes = plt.subplots(nrows, ncols, figsize=(width*ncols, height)) 
        self.fig  = fig 
        self.axes = axes
        self.filename = filename
        self.ax2img = {}
        self.ax2cb  = {}

    def __enter__(self):
        # Set up the writer
        self.writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Me'), bitrate=1800)
        self.writer.setup(self.fig, self.filename, dpi=100)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Finish the animation
        self.writer.finish()
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            return False

    def grab_frame(self, **savefig_kwargs):
        # Grab the current frame
        self.writer.grab_frame(**savefig_kwargs)

    def update(self):
        self.grab_frame()

    def draw_mesh(self, mesh, value, ax=None, show_colorbar=True, title=None, update=True, show_mesh=True, umin=None, umax=None):
        if ax is None:
            assert not isinstance(self.axes, np.ndarray), "ax must be specified when there are multiple axes"
            ax = self.axes 
        if ax in self.ax2img:
            img = self.ax2img[ax]
            img.set_array(value)
            if umin is not None and umax is not None:
                img.set_clim(umin, umax)
            if show_colorbar:
                self.ax2cb[ax].update_normal(img)
        else:
            img = draw_mesh(mesh.points, 
                    mesh.elements(), 
                    value, 
                    ax=ax, 
                    show_colorbar=show_colorbar, 
                    show_mesh=show_mesh)
            if show_colorbar:
                img, cb = img
                self.ax2img[ax] = img
                self.ax2cb[ax] = cb
                if umin is not None and umax is not None:
                    img.set_clim(umin, umax)
                    cb.update_normal(img)
            else:
                self.ax2img[ax] = img
                if umin is not None and umax is not None:
                    img.set_clim(umin, umax)

        if title is not None:
            ax.set_title(title)
        if update:
            self.grab_frame()

class Mesh(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        # turn is_... or ..._mask to bool
        for key in list(mesh.point_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.point_data[key] = mesh.point_data[key].astype(bool)
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                if key.startswith("is_") or key.endswith("_mask"):
                    mesh.cell_data[key][i] = _v.astype(bool)
        for key in list(mesh.field_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.field_data[key] = mesh.field_data[key].astype(bool)
        
        # cells
        self.cells  = BufferDict({k:torch.from_numpy(v).long() for k,v in mesh.cells_dict.items()})
        
        # point data
        self.point_data = BufferDict({k:torch.from_numpy(v) for k,v in mesh.point_data.items()})

        # cell data
        self.cell_data  = BufferDict({
            k:BufferDict({i:torch.from_numpy(_v) for i,_v in v.items()}) for k,v in mesh.cell_data_dict.items()
        })
   
        # field data
        self.field_data = BufferDict({k:torch.from_numpy(v) for k,v in mesh.field_data.items()})

        # cell setes useless
        self.cell_sets = mesh.cell_sets

        self.dim2eletyp = defaultdict(list) # Dict[int, List[str]]
        for element_type in self.cells.keys():
            self.dim2eletyp[topological_dimension[element_type]].append(element_type)
        self.default_eletyp = self.dim2eletyp[max(self.dim2eletyp.keys())] 
        if len(self.default_eletyp) == 1: # if only one element type, use it as default
            self.default_eletyp = self.default_eletyp[0]

        dimension = max(self.dim2eletyp.keys())

        self.register_buffer(
            "points",
            torch.from_numpy(mesh.points[:, :dimension])
        )

    def register_point_data(self, key, value):
        assert key not in self.point_data.keys(), f"the key {key} already exists in point_data"
        assert value.shape[0] == self.points.shape[0], f"the first dimension of value should be {self.points.shape[0]}, but got {value.shape[0]}"
        self.point_data.register_buffer(key, value)

        return self

    def to_meshio(self):
        
        mesh = meshio.Mesh(
            points = self.points.detach().cpu().numpy(),
            cells  = {k:v.detach().cpu().numpy() for k,v in self.cells.items()},
            point_data = {k:v.detach().cpu().numpy() for k,v in self.point_data.items()},
            cell_data  = {k:[_v.detach().cpu().numpy() for _v in v.values()] for k,v in self.cell_data.items()},
            field_data = {k:v.detach().cpu().numpy() for k,v in self.field_data.items()},
            cell_sets = self.cell_sets
        )  
        return mesh

    def save(self, file_name:str, file_format:str=None):
        mesh = self.to_meshio()
        # turn is_... or ..._mask to float
        for key in list(mesh.point_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.point_data[key] = mesh.point_data[key].astype(float)
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                if key.startswith("is_") or key.endswith("_mask"):
                    mesh.cell_data[key][i] = _v.astype(float)
        for key in list(mesh.field_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.field_data[key] = mesh.field_data[key].astype(float)
        
        # assert no bool variables, since file cannot save bool
        for key in list(mesh.point_data.keys()):
            assert mesh.point_data[key].dtype != bool, f"PointData: bool is not supported in meshio, but got {key}"
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                assert _v.dtype != bool, f"CellData: bool is not supported in meshio, but got {key}"
        for key in list(mesh.field_data.keys()):
            assert mesh.field_data[key].dtype != bool, f"FieldData: bool is not supported in meshio, but got {key}"
        
        if file_name.endswith(".vtk") or file_name.endswith(".vtu"):
            # if vtk/vtu turn 2d to 3d 
            if mesh.points.shape[1] == 2:
                mesh.points = np.concatenate([mesh.points, torch.zeros(mesh.points.shape[0], 1)], -1)
            if "u" not in mesh.point_data.keys():
                mesh.point_data["u"] = np.zeros((mesh.points.shape[0], )) 
         
        meshio.write(file_name, mesh, file_format)
        return self

    def to_file(self, file_name:str, file_format:str=None):
        return self.save(file_name, file_format)

    def elements(self, element_type=None):
        if element_type is None:
            element_type = self.default_eletyp
        if isinstance(element_type, str):
            return self.cells[element_type]
        elif isinstance(element_type, Iterable):
            return {k:self.cells[k] for k in element_type}
        else:
            raise Exception(f"element_type must be str or Iterable[str], but got {element_type}")
    
    def clone(self):
        return Mesh(self.to_meshio())

    def plot(self, kwargs= None, save_path=None, backend="matplotlib", dt=None, show_mesh=False, fix_clim=False):
        
        if kwargs is None:
            elements = mesh.elements()
            points   = mesh.points.cpu().numpy()

            assert points.shape[1] == 2, f"points must be 2D, but got {points.shape}"

            fig, ax = plt.subplots(figsize=(8,  8))

            def draw_elements(elements):
                if elements.shape[1] == 3: # tri
                    ax.triplot(points[:,0], points[:,1], elements, color='k', linewidth=0.5)
                elif elements.shape[1] == 4: # quad
                    polygons = [patches.Polygon(points[element], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
                    polygons = PatchCollection(polygons, match_original=True)
                    ax.add_collection(polygons)
                elif elements.shape[1] == 6: # tri6
                    order = np.array([0, 3, 1, 4, 2, 5])
                    polygons = [patches.Polygon(points[element[order]], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
                    polygons = PatchCollection(polygons, match_original=True)
                    ax.add_collection(polygons)
                elif elements.shape[1] == 9: # quad9 
                    order = np.array([0, 4, 1, 5, 2, 6, 3, 7])
                    polygons = [patches.Polygon(points[element[order]], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
                    polygons = PatchCollection(polygons, match_original=True)
                    ax.add_collection(polygons)
                else:
                    raise NotImplementedError(f"element type {elements.shape[1]} is not supported")
        
            if isinstance(elements, torch.Tensor):
                elements = elements.detach().cpu().numpy()
                draw_elements(elements)
            elif isinstance(elements, dict):
                for value in elements.values():
                    draw_elements(value.detach().cpu().numpy())
            else:
                raise NotImplementedError(f"elements type {type(elements)} is not supported")
            
            ax.scatter(points[:,0], points[:,1], s=1, c='orange')
            ax.axis("equal")
            ax.axis("off")
            if save_path is None:
                plt.show()
            else:
                fig.savefig(save_path, dpi=400)

        else:

            points = mesh.points
            elements = mesh.elements()
        
            ncols = len(kwargs.keys())
            width = mesh.points[:,0].max() - mesh.points[:,0].min()
            height = mesh.points[:,1].max() - mesh.points[:,1].min()
            ratio  = (width / height).item()
            fig, ax = plt.subplots(1, ncols, figsize=(5*ncols * ratio, 5))
            key, value = next(iter(kwargs.items()))
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            if isinstance(elements, torch.Tensor):
                elements = elements.detach().cpu().numpy()
            if not isinstance(ax,  np.ndarray):
                ax = [ax]
            if isinstance(value,(torch.Tensor, np.ndarray)):      
                if save_path is None:
                    save_path = 'mesh.png'
                for i, (key, value) in enumerate(kwargs.items()):
                    img, cb = draw_mesh(points, elements, value, ax=ax[i], show_colorbar=True, show_mesh=show_mesh)
                    ax[i].set_title(key)
                fig.savefig(save_path, dpi=400)
            elif isinstance(value, (list, tuple)):
                if save_path is None:
                    save_path = 'mesh.gif'
                if fix_clim:
                    vmin = min([v.min() for v in value])
                    vmax = max([v.max() for v in value])
                cbs = []
                imgs = []
                for i, (key, value) in enumerate(kwargs.items()):
                    img,cb = draw_mesh(points, elements, value[0], ax=ax[i], show_colorbar=True,show_mesh=show_mesh)
                    if fix_clim:
                        img.set_clim(vmin, vmax)
                        cb.update_normal(img)
                    ax[i].set_title(key)
                    cbs.append(cb)
                    imgs.append(img)
                if dt is not None:
                    fig.suptitle(f"t={0*dt:7.5f}")
                else:
                    fig.suptitle(f"Frame:{0:5d}")
                def update(frame):
                    for i, (key, value) in enumerate(kwargs.items()):
                        v   = value[frame].detach().cpu().numpy() if isinstance(value[frame], torch.Tensor) else value[frame]
                        if not fix_clim:   
                            imgs[i].set_clim(v.min(), v.max())
                        imgs[i].set_array(v)
                        if not fix_clim:
                            cbs[i].update_normal(imgs[i])
                    if dt is not None:
                        fig.suptitle(f"t={frame*dt:7.5f}")
                    else:
                        fig.suptitle(f"Frame:{frame:5d}")
                    return imgs
                anim = FuncAnimation(fig, update, frames=len(value), interval=100)
                anim.save(save_path, fps=10,  dpi=400)

    @property
    def n_point(self):
        return self.points.shape[0]

    @property
    def boundary_mask(self):
        if "is_boundary" in self.point_data.keys():
            return self.point_data["is_boundary"]
        elif "boundary_mask" in self.point_data.keys():
            return self.point_data["boundary_mask"]
        else:
            raise Exception("'boundary_mask' or 'is_boundary' is not found in point_data")

    @property
    def default_element_type(self):
        return self.default_eletyp

    @property
    def dtype(self):
        return self.points.dtype
    
    @property 
    def device(self):
        return self.points.device

    @classmethod
    def from_meshio(cls,mesh):
        return cls(mesh)
    
    @classmethod
    def read(cls, file_name:str, file_format:str=None):
        return cls(meshio.read(file_name, file_format))
    
    @classmethod
    def from_file(cls, file_name:str, file_format:str=None):
        return cls.read(file_name, file_format)

    @staticmethod
    def gen_rectangle(chara_length=0.1,
             order=1,
             element_type="tri",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
             visualize=False,
             cache_path=None):
        assert left < right, f"left must be smaller than right, but got {left} >= {right}"
        assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
        assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
        assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

        if cache_path is None:
            cache_path = f".gmsh_cache/rectangle_{left}_{right}_{bottom}_{top}_{chara_length}_{order}_{element_type}.msh"

        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):

            width, height = right - left, top - bottom

            gmsh.initialize()
            gmsh.model.add("rectangle")

            rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)

            gmsh.model.occ.synchronize()

            if element_type == "quad":
                # Set transfinite meshing
                gmsh.model.mesh.setTransfiniteSurface(rectangle, "Right")
                # Apply the recombine algorithm to generate quad elements
                gmsh.model.mesh.setRecombine(2, rectangle)

            # Set the element order to 2 to generate second-order elements
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

            gmsh.model.addPhysicalGroup(2, [rectangle])
            gmsh.model.setPhysicalName(2, 1, "domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

            if visualize:
                gmsh.fltk.run()

            # Save the mesh
            gmsh.write(cache_path)

            # Finalize Gmsh
            gmsh.finalize()

        mesh = Mesh.from_file(cache_path)

        is_left_boundary  = mesh.points[:, 0] == left
        is_right_boundary = mesh.points[:, 0] == right
        is_bottom_boundary= mesh.points[:, 1] == bottom
        is_top_boundary   = mesh.points[:, 1] == top
        is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary
        mesh.register_point_data("is_boundary", is_boundary)
        mesh.register_point_data("is_left_boundary", is_left_boundary)
        mesh.register_point_data("is_right_boundary", is_right_boundary)
        mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
        mesh.register_point_data("is_top_boundary", is_top_boundary)

        return mesh
    
    @staticmethod
    def gen_hollow_rectangle(
        chara_length=0.1,
        order=1,
        element_type="quad",
        outer_left=0.0, outer_right=1.0, outer_bottom=0.0, outer_top=1.0,
        inner_left = 0.25,  inner_right=0.75,
        inner_bottom =0.25, inner_top=0.75,
        visualize=False,
        cache_path=None
    ):
        assert inner_left < inner_right, f"inner_left must be smaller than inner_right, but got {inner_left} >= {inner_right}"
        assert inner_bottom < inner_top, f"inner_bottom must be smaller than inner_top, but got {inner_bottom} >= {inner_top}"
        assert inner_left > outer_left, f"inner_left must be greated than left, but got {inner_left} <= {outer_left}"
        assert inner_right < outer_right, f"inner_right must be smaller than right, but got {inner_right} >= {outer_right}"
        assert inner_bottom > outer_bottom, f"inner_bottom must be greater than bottom, but  got {inner_bottom} <= {outer_bottom}"
        assert inner_top < outer_top, f"inner_top must be smaller than outer_top, but got {inner_top} > {outer_top}"
        assert outer_left < outer_right, f"left must be smaller than right, but got {outer_left} >= {outer_right}"
        assert outer_bottom < outer_top, f"outer_bottom must be smaller than outer_top, but got {outer_bottom} >= {outer_top}"
        assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
        assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

        if cache_path is None:
            cache_path = f".gmsh_cache/hollow_rectangle_{outer_left}_{outer_right}_{outer_bottom}_{outer_top}_{inner_left}_{inner_right}_{inner_bottom}_{inner_top}_{chara_length}_{order}_{element_type}.msh"
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):

            width, height = outer_right - outer_left, outer_top - outer_bottom
            inner_width, inner_height = inner_right - inner_left, inner_top - inner_bottom
            gmsh.initialize()
            gmsh.model.add("rectangle")

            rectangle_outer = gmsh.model.occ.addRectangle(outer_left, outer_bottom, 0, width, height)
            rectangle_inner = gmsh.model.occ.addRectangle(inner_left, inner_bottom, 0, inner_width, inner_height)

            gmsh.model.occ.synchronize()

            _ = gmsh.model.occ.cut([(2,rectangle_outer)], [(2,rectangle_inner)])

            gmsh.model.occ.synchronize()

            if element_type == "quad":
                # Set transfinite meshing
                # gmsh.model.mesh.setTransfiniteSurface(rectangle, "Right")
                # Apply the recombine algorithm to generate quad elements
                gmsh.model.mesh.setRecombine(2, rectangle_outer)

            # Set the element order to 2 to generate second-order elements
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

            gmsh.model.addPhysicalGroup(2, [rectangle_outer])
            gmsh.model.setPhysicalName(2, 1, "domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

            if visualize:
                gmsh.fltk.run()

            # Save the mesh
            gmsh.write(cache_path)

            # Finalize Gmsh
            gmsh.finalize()

        mesh = Mesh.from_file(cache_path)

        is_outer_left_boundary  = mesh.points[:, 0] == outer_left
        is_outer_right_boundary = mesh.points[:, 0] == outer_right
        is_outer_bottom_boundary= mesh.points[:, 1] == outer_bottom
        is_outer_top_boundary   = mesh.points[:, 1] == outer_top
        is_inner_left_boundary   = mesh.points[:,0] == inner_left
        is_inner_right_boundary  = mesh.points[:,0] == inner_right 
        is_inner_bottom_boundary = mesh.points[:,1] == inner_bottom
        is_inner_top_boundary    = mesh.points[:,1] == inner_top
        is_outer_boundary       = is_outer_left_boundary | is_outer_right_boundary | is_outer_bottom_boundary | is_outer_top_boundary
        is_inner_boundary       = is_inner_left_boundary | is_inner_right_boundary | is_inner_bottom_boundary | is_inner_top_boundary
        is_boundary             = is_inner_boundary | is_outer_boundary
        mesh.register_point_data("is_boundary", is_boundary)
        mesh.register_point_data("is_inner_left_boundary", is_inner_left_boundary)
        mesh.register_point_data("is_outer_left_boundary", is_outer_left_boundary)
        mesh.register_point_data("is_inner_right_boundary", is_inner_right_boundary)
        mesh.register_point_data("is_outer_right_boundary", is_outer_right_boundary)
        mesh.register_point_data("is_inner_bottom_boundary", is_inner_bottom_boundary)
        mesh.register_point_data("is_outer_bottom_boundary", is_outer_bottom_boundary)
        mesh.register_point_data("is_inner_top_boundary", is_inner_top_boundary)
        mesh.register_point_data("is_outer_top_boundary", is_outer_top_boundary)

        return mesh

    @staticmethod
    def gen_circle(chara_length=0.1,
            order=1,
            element_type="tri",
            cx = 0.0, cy = 0.0, r = 1.0,
            visualize=False,
            cache_path=None):
        assert r > 0, f"r must be positive, but got {r} <= 0"
        assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
        assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

        if cache_path is None:
            cache_path = f".gmsh_cache/circle_{cx}_{cy}_{r}_{chara_length}_{order}_{element_type}.msh"

        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):

            gmsh.initialize()
            gmsh.model.add("Circle")

            circle = gmsh.model.occ.addDisk(cx, cy, 0, r)

            gmsh.model.occ.synchronize()

            if element_type == "quad":
                # Set transfinite meshing
                gmsh.model.mesh.setTransfiniteSurface(circle, "Right")
                # Apply the recombine algorithm to generate quad elements
                gmsh.model.mesh.setRecombine(2, circle)

            # Set the element order to 2 to generate second-order elements
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

            gmsh.model.addPhysicalGroup(2, [circle])
            gmsh.model.setPhysicalName(2, 1, "domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

            if visualize:
                gmsh.fltk.run()

            # Save the mesh
            gmsh.write(cache_path)

            # Finalize Gmsh
            gmsh.finalize()

        mesh = Mesh.from_file(cache_path)

        radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2)
        is_boundary = radius == r
        mesh.register_point_data("is_boundary", is_boundary)

        return mesh
    
    @staticmethod
    def gen_hollow_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        assert r_inner > 0, f"r_inner must be positive, but got {r_inner} <= 0"
        assert r_outer > 0, f"r_outer must be positive, but got {r_outer} <= 0"
        assert r_outer > r_inner, f"r_outer must be greater than r_inner, but got {r_outer} <= {r_inner}"
        assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
        assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"
    
        if cache_path is None:
            cache_path = f".gmsh_cache/circle_{cx}_{cy}_{r_inner}_{r_outer}_{chara_length}_{order}_{element_type}.msh"

        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):

            gmsh.initialize()
            gmsh.model.add("HollowCircle")

            circle_inner = gmsh.model.occ.addDisk(cx, cy, 0, r_inner, r_inner,)
            circle_outer = gmsh.model.occ.addDisk(cx, cy, 0, r_outer, r_outer)

            gmsh.model.occ.synchronize()

            hollow_entity, _ = gmsh.model.occ.cut([(2, circle_outer)], [(2, circle_inner)])
            hollow_circle = hollow_entity[0][-1]
            gmsh.model.occ.synchronize()

            if element_type == "quad":
                # Set transfinite meshing
                # gmsh.model.mesh.setTransfiniteSurface(circle_outer, "Right")
                # Apply the recombine algorithm to generate quad elements
                gmsh.model.mesh.setRecombine(2, circle_outer)

            # Set the element order to 2 to generate second-order elements
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

            gmsh.model.addPhysicalGroup(2, [circle_inner])
            gmsh.model.setPhysicalName(2, 1, "domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

            if visualize:
                gmsh.fltk.run()

            # Save the mesh
            gmsh.write(cache_path)

            # Finalize Gmsh
            gmsh.finalize()

        mesh = Mesh.from_file(cache_path)

        radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2)
        is_inner_boundary = torch.isclose(radius, torch.ones_like(radius) * r_inner)
        is_outer_boundary = torch.isclose(radius, torch.ones_like(radius) * r_outer)
        is_boundary = is_inner_boundary | is_outer_boundary
        mesh.register_point_data("is_inner_boundary", is_inner_boundary)
        mesh.register_point_data("is_outer_boundary", is_outer_boundary)
        mesh.register_point_data("is_boundary", is_boundary)

        return mesh

    @staticmethod
    def gen_L(chara_length=0.1,
             order=1,
             element_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0, 
             top_inner=0.5,
             right_inner=0.5,
             visualize=False,
             cache_path=None):
        assert left < right, f"left must be smaller than right, but got {left} >= {right}"
        assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
        assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
        assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

        if cache_path is None:
            cache_path = f".gmsh_cache/L_{left}_{right}_{bottom}_{top}_{top_inner}_{right_inner}_{chara_length}_{order}_{element_type}.msh"

        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        if not os.path.exists(cache_path):

            width, height = right - left, top - bottom

            gmsh.initialize()
            gmsh.model.add("rectangle")

            rectangle_outer = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
            rectangle_inner = gmsh.model.occ.addRectangle(right_inner, top_inner, 0, right-right_inner, top-top_inner)

            gmsh.model.occ.synchronize()

            _ = gmsh.model.occ.cut([(2,rectangle_outer)], [(2,rectangle_inner)])

            gmsh.model.occ.synchronize()

            if element_type == "quad":
                # Set transfinite meshing
                # gmsh.model.mesh.setTransfiniteSurface(rectangle_outer, "Right")
                # Apply the recombine algorithm to generate quad elements
                gmsh.model.mesh.setRecombine(2, rectangle_outer)

            # Set the element order to 2 to generate second-order elements
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

            gmsh.model.addPhysicalGroup(2, [rectangle_outer])
            gmsh.model.setPhysicalName(2, 1, "domain")

            # Generate the mesh
            gmsh.model.mesh.generate(2)

            if visualize:
                gmsh.fltk.run()

            # Save the mesh
            gmsh.write(cache_path)

            # Finalize Gmsh
            gmsh.finalize()

        mesh = Mesh.from_file(cache_path)

        is_left_boundary  = mesh.points[:, 0] == left
        is_right_boundary = mesh.points[:, 0] == right
        is_bottom_boundary= mesh.points[:, 1] == bottom
        is_top_boundary   = mesh.points[:, 1] == top
        is_L_top_boundary = mesh.points[:, 1] == top_inner
        is_L_right_boundary = mesh.points[:, 0] == right_inner
        is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary | is_L_top_boundary | is_L_right_boundary
        mesh.register_point_data("is_boundary", is_boundary)
        mesh.register_point_data("is_left_boundary", is_left_boundary)
        mesh.register_point_data("is_right_boundary", is_right_boundary)
        mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
        mesh.register_point_data("is_top_boundary", is_top_boundary)
        mesh.register_point_data("is_L_top_boundary", is_L_top_boundary)
        mesh.register_point_data("is_L_right_boundary", is_L_right_boundary)

        return mesh

class Condenser:

    def __init__(self, dirichlet_mask:torch.Tensor, dirichlet_value:torch.Tensor = None):
        assert dirichlet_mask.dtype == torch.bool, "the dtype of dirichlet_mask must be torch.bool"
        assert dirichlet_mask.ndim == 1, "tDirichlet_mask must be 1D tensor"
        assert dirichlet_value is None or dirichlet_value.ndim == 1, "dirichlet_value must be 1D tensor"
        self.dirichlet_mask  = dirichlet_mask
        if dirichlet_value is None:
            self.dirichlet_value = torch.zeros(self.dirichlet_mask.sum())
        elif dirichlet_value.shape[0] == dirichlet_mask.shape[0]:
            self.dirichlet_value = dirichlet_value[self.dirichlet_mask]
        else:
            assert dirichlet_value.shape[0] == dirichlet_mask.sum(), "the shape of dirichlet_value must be [n_dof] or [n_outer_dof]"
            self.dirichlet_value = dirichlet_value
        

        self.inner_row = None
        self.inner_col = None
        self.ou2in_row = None
        self.ou2in_col = None
        self.is_inner_edge = None
        self.is_ou2in_edge = None
        self.layout_hash   = None
        self.K_ou2in       = None

    def _compute_layout(self, matrix:SparseMatrix):
        edge_u, edge_v               = matrix.row, matrix.col
        n_dof                        = matrix.shape[0]

        is_inner_dof, is_outer_dof = ~self.dirichlet_mask, self.dirichlet_mask
        
        is_inner_u,    is_inner_v    = is_inner_dof[edge_u], is_inner_dof[edge_v]
        is_outer_u,    is_outer_v    = is_outer_dof[edge_u], is_outer_dof[edge_v]
        is_inner_edge, is_ou2in_edge = is_inner_u & is_inner_v, is_inner_u & is_outer_v
        n_inner_dofs, n_outer_dofs = is_inner_dof.sum().item(), is_outer_dof.sum().item()
        local_nids = torch.full((n_dof,), -1, dtype=torch.long, device=matrix.device)
        local_nids[is_inner_dof] = torch.arange(n_inner_dofs, device=matrix.device)
        local_nids[is_outer_dof] = torch.arange(n_outer_dofs, device=matrix.device)

        self.inner_row = local_nids[edge_u[is_inner_edge]]
        self.inner_col = local_nids[edge_v[is_inner_edge]]
        self.ou2in_row = local_nids[edge_u[is_ou2in_edge]]
        self.ou2in_col = local_nids[edge_v[is_ou2in_edge]]
        self.is_inner_edge = is_inner_edge
        self.is_ou2in_edge = is_ou2in_edge
        self.inner_shape = (n_inner_dofs, n_inner_dofs)
        self.ou2in_shape = (n_inner_dofs, n_outer_dofs)
        self.layout_hash = matrix.layout_hash
        self.n_inner_dof = n_inner_dofs
        self.n_outer_dof = n_outer_dofs
        self.n_dof       = n_dof
        self.is_inner_dof = is_inner_dof
        self.is_outer_dof = is_outer_dof

    def __call__(self, matrix:SparseMatrix, rhs:torch.Tensor = None):
        if rhs is None:
            rhs = torch.zeros(matrix.shape[0])
       
        if self.inner_row is None:
            self._compute_layout(matrix)

        assert matrix.shape[0] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.shape[1] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.has_same_layout(self.layout_hash), "the layout of the matrix is changed, please recompute the condensed matrix"
        assert rhs.ndim == 1, "rhs must be 1D tensor"
        assert rhs.shape[0] == self.n_dof, f"the shape of rhs must be [{self.n_dof}], but got {rhs.shape}"
        
        K_inner = SparseMatrix(
            matrix.edata[self.is_inner_edge], self.inner_row, self.inner_col, self.inner_shape, 
        )
        K_ou2in = SparseMatrix(
            matrix.edata[self.is_ou2in_edge], self.ou2in_row, self.ou2in_col, self.ou2in_shape, 
        )
        self.K_ou2in = K_ou2in

        self.dirichlet_value = self.dirichlet_value.type(K_inner.edata.dtype).to(K_inner.edata.device)
        rhs  = rhs.type(K_inner.edata.dtype).to(K_inner.edata.device)
       
        return K_inner, rhs[self.is_inner_dof] - K_ou2in @ self.dirichlet_value

    def condense_rhs(self, rhs):
        assert self.K_ou2in is not None, f"please call __call__ first"

        self.dirichlet_value = self.dirichlet_value.type(rhs.dtype).to(rhs.device)
        rhs = rhs.type(self.K_ou2in.edata.dtype).to(self.K_ou2in.edata.device)

        return rhs[self.is_inner_dof] - self.K_ou2in @ self.dirichlet_value
       
    def recover(self, u:torch.Tensor):
        assert u.ndim == 1, "u must be 1D tensor"
        assert u.shape[0] == self.n_inner_dof, f"the shape of u must be [{self.n_inner_dof}], but got {u.shape}"

        u_full = torch.zeros(self.n_dof, dtype=u.dtype, device=u.device)
        u_full[self.is_inner_dof] += u 
        u_full[self.is_outer_dof] += self.dirichlet_value

        return u_full
    

def dot(a, b, reduce_dim=-1):
    if reduce_dim == -1:
        return torch.einsum("...ik,...jk->...ij", a, b)
    elif reduce_dim == -2:
        return torch.einsum("...ika,...jkb->...ijab", a, b)
    else:
        raise ValueError(f"reduce_dim must be -1 or -2, but got {reduce_dim}")

def mul(a, b):
    return torch.einsum("...i,...j->...ij", a, b)

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv, c):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
                c: torch.Tensor[]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return dot(gradu, gradv) * c * c
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return mul(u, v)
    
class Wave:
    def __init__(self, mesh):
        self.M_asm = MAssembler.from_mesh(mesh)
        self.A_asm = AAssembler.from_assembler(self.M_asm)
        self.mesh = mesh
        self.condenser = Condenser(mesh.point_data['is_bottom_boundary'])

    def __call__(self, u0, c, dt=0.001, n=100):
        """
        Parameters
        ----------
            u0: torch.Tensor[n_point]
                initial condition
            c: torch.Tensor[n_point]
                wave speed
            dt: float, default 0.001
                time interval
            n: int,  default 100
                number of time steps
        Returns:
        --------
            Us: torch.Tensor[n, n_point]
        """

        M = self.M_asm(mesh.points)
        A = self.A_asm(mesh.points, point_data={"c":c})

        Us  = [u0] 
        v0 = torch.zeros_like(u0) 
        A = A
        K = 2 * M
        F = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
        K_, F_ = self.condenser(K, F)
        U_     = K_.solve(F_)
        U      = self.condenser.recover(U_)
       
        M_     = self.condenser(M)[0]


        Us.append(U)
        for _ in range(n-2):
            U1, U2 = Us[-2:]

            F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
            
            F_ = self.condenser.condense_rhs(F)

            U_ = M_.solve(F_)

            U  = self.condenser.recover(U_)

            Us.append(U)

        return torch.stack(Us, 0)


def circle_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    # ground truth c
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 
    c_gt[(x-0.5)**2+(y-0.5)**2 < 0.2**2] = 2.0 
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh 

def circles_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    # ground truth c
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 
    c_gt[(x-0.7)**2+(y-0.2)**2 < 0.1**2] = 2.0 
    c_gt[(x-0.2)**2+(y-0.7)**2 < 0.1**2] = 2.0
    c_gt[(x-0.6)**2+(y-0.7)**2 < 0.1**2] = 2.0
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh 

def eth_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    image = Image.open('eth.png')
    # get the height and width of the image
    image = np.array(image)
    alpha = image[:,:,3]
    ratio = alpha.shape[1] / alpha.shape[0]
    mesh = Mesh.gen_rectangle(right=ratio, chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 

    y,x = mesh.points.T.cpu().numpy() * alpha.shape[0]
    x    = alpha.shape[0] - x
    coord = np.vstack((x,y))
    alpha_points = torch.from_numpy(scipy.ndimage.map_coordinates(alpha, coord, mode="nearest")).type(mesh.points.dtype).to(mesh.points.device)
    c_gt[alpha_points >0] = 2.0
  

    x, y = mesh.points[:,0], mesh.points[:,1]
    x_source, y_source = ratio/2, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n', type=int, default=100)
    parser.add_argument('-dt','--dt', type=float, default=1e-2)
    parser.add_argument('-nd','--n_detector', type=int, default=100)
    parser.add_argument('-s','--sigma', type=float, default=0.1)
    parser.add_argument('-A','--A', type=float, default=10.0)
    parser.add_argument('-e','--epoch', type=int, default=400)
    parser.add_argument('--eval_every_eps', type=int, default=10)
    parser.add_argument("-d","--chara_length", type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--detect_mode', type=str, default="all", choices=["all", "top"])
    parser.add_argument('--dataset', type=str, default="circles", choices=['circle','circles','eth'])
    parser.add_argument('--optimizer', type=str,default="adam", choices=["adam", "lbfgs"])
    
    args = parser.parse_args()
    dt = args.dt
    n  = args.n
    n_detector = args.n_detector
    sigma = args.sigma
    A     = args.A
    epoch = args.epoch
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    torch.random.manual_seed(123456)
    
    mesh = {
        "circle":circle_dataset,
        "circles":circles_dataset,
        "eth":eth_dataset
    }[args.dataset](args.chara_length, device=device, A=A, sigma=sigma)
    top_idx = torch.where(mesh.point_data['is_top_boundary'])[0]
    all_idx = torch.arange(mesh.n_point, device=device)
    candidiate_idx = top_idx if args.detect_mode == "top" else all_idx
    n_detector = min(n_detector, len(candidiate_idx))
    sample_idx = random.sample(range(len(candidiate_idx)), n_detector)
    print(f"select portion:{n_detector/len(candidiate_idx)}")
    detector_idx = candidiate_idx[sample_idx]
    dbc_constraint = mesh.point_data['is_bottom_boundary']
    
    wave = Wave(mesh)


    c_gt  = mesh.point_data['c_gt']
    u0    = mesh.point_data['u0']
    us_gt = wave(u0, c_gt, dt, n)

    # prediction c
    c_pred = torch.ones_like(c_gt).requires_grad_(True) 
    loss_fn = torch.nn.MSELoss()
   
    cs_pred = []
    losses  = []
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([c_pred], lr=0.1, max_iter=50000, line_search_fn="strong_wolfe", tolerance_change=1e-10)
        pbar = tqdm(total = 50000)

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam([c_pred], lr=args.lr)
        pbar = tqdm(total = epoch)

    else:
        raise NotImplementedError(f"optimizer {args.optimizer} not implemented")
    
    def closure():
        optimizer.zero_grad()
        us_pred = wave(u0, c_pred, dt, n)
        loss = loss_fn(us_pred[:, detector_idx], us_gt[:, detector_idx])
        loss.backward()
        pbar.set_postfix({"loss":loss.item()})
        cs_pred.append(c_pred.detach().clone())
        losses.append(loss.item())
        pbar.update(1)
        return loss
  
    if args.optimizer == "lbfgs":
        optimizer.step(closure)
    else:
        for i in range(epoch):
            closure()
            optimizer.step()

    with torch.no_grad():
        us_pred = wave(u0, c_pred, dt, n)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.savefig("c_loss.png", dpi=400)
    # breakpoint()
    mesh.plot({"prediction":[us_pred[i] for i in range(len(us_pred))], "ground truth":[us_gt[i] for i in range(len(us_gt))]},
              save_path="c_compare.mp4", dt=dt, show_mesh=True, fix_clim=False)


    width = mesh.points[:,0].max() - mesh.points[:,0].min()
    height = mesh.points[:,1].max() - mesh.points[:,1].min()
    ratio = (width / height).item()
    with StreamPlotter(ncols=2, width=ratio*5,height=5, filename="c_optimization.mp4") as sp:
        sp.draw_mesh(mesh, c_gt, ax=sp.axes[0], title="ground truth", update=False)
        for i, c_pred in enumerate(cs_pred):
            if i % args.eval_every_eps == 0:
                sp.draw_mesh(mesh, c_pred, ax=sp.axes[1], title=f"epoch {i}", update=False, show_mesh=False)
                sp.axes[1].scatter(mesh.points[detector_idx,0], mesh.points[detector_idx,1], c="r", label="detector", s=2)
                if i == 0:
                    sp.axes[1].legend()
                sp.update()

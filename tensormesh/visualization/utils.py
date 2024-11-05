from re import T
from matplotlib.pyplot import isinteractive
import torch 
import numpy as np
from functools import lru_cache
from typing import Sequence, Union, TypeVar,Generic
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dia_matrix, dok_matrix, lil_matrix, issparse
from ..sparse import SparseMatrix
ScipySparseMatrix = Union[coo_matrix, csr_matrix, csc_matrix, dia_matrix, dok_matrix, lil_matrix]
def as_sparse_matrix(x:Union[SparseMatrix,ScipySparseMatrix])->SparseMatrix:
    if issparse(x):
        x = x.tocoo()
        x = SparseMatrix.from_scipy_coo(x)
    elif isinstance(x, SparseMatrix):
        x = x.detach().cpu()
    else:
        raise TypeError(f"{type(x)} is not acceptable for SparseMatrix|ScipySparseMatrix")
    return x

def as_tensor(x:Union[torch.Tensor,np.ndarray])->torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        raise ValueError(f"unsupported type {type(x)}")


def as_ndarray(x:Union[torch.Tensor,np.ndarray])->np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"unsupported type {type(x)}")

def dim(x:torch.Tensor|np.ndarray)->int:
    if isinstance(x, torch.Tensor):
        return x.dim()
    elif isinstance(x, np.ndarray):
        return len(x.shape)
    else:
        raise ValueError(f"unsupported type {type(x)}")
    
@lru_cache()
def grid(dim:int, min_vals:Sequence[float], max_vals:Sequence[float], density:int=100) -> np.ndarray:
    """Create a grid of points in 2D or 3D space.

    Parameters
    ----------
    dim : int
        Dimension of the grid (2 or 3)
    min_vals : Sequence[float]
        Minimum values for each dimension
    max_vals : Sequence[float] 
        Maximum values for each dimension
    density : int, optional
        Number of points along each dimension, by default 100

    Returns
    -------
    np.ndarray
        Grid points with shape (density^dim, dim)
    """
    assert dim in [2,3], f"dim must be 2 or 3, got {dim}"
    assert len(min_vals) == dim, f"min_vals must have length {dim}"
    assert len(max_vals) == dim, f"max_vals must have length {dim}"

    if dim == 2:
        x = np.linspace(min_vals[0], max_vals[0], density)
        y = np.linspace(min_vals[1], max_vals[1], density)
        X, Y = np.meshgrid(x, y)
        return np.column_stack((X.flatten(), Y.flatten()))
    else:
        x = np.linspace(min_vals[0], max_vals[0], density)
        y = np.linspace(min_vals[1], max_vals[1], density)
        z = np.linspace(min_vals[2], max_vals[2], density)
        X, Y, Z = np.meshgrid(x, y, z)
        return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
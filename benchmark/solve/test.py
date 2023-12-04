import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
sys.path.append("../..")
import torch 
from torch_fem.sparse.solve.torch_solve import bicgstab, cg
def create_sparse_system(n):
    """ Create a sparse linear system for testing. """
  
    # A = scipy.sparse.random(n, n, density=0.001, format='coo')
    # A = A + A.T
    # A = A + scipy.sparse.eye(n) * 0.01
        
    # u = np.random.rand(n)
    # b = A @ u

    # A = A.tocoo()
    # return A, b
    A = scipy.sparse.load_npz("tmp.npz")
    b = np.load("tmp.npy")
    return A, b
def solve_torch(A, b):
    """ Solve using torch """
    x = cg(A, b)

A, b = create_sparse_system(1000)
A = A.tocsr()
A_th = torch.sparse_csr_tensor(
    torch.from_numpy(A.indptr), torch.from_numpy(A.indices), torch.from_numpy(A.data), torch.Size(A.shape))
b_th = torch.from_numpy(b)
solve_torch(A_th, b_th)
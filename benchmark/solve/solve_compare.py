
import cupyx.scipy.sparse.linalg
import cupy as cp
import pandas as pd

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from petsc4py import PETSc
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sympy import solve
from tqdm import tqdm
import tracemalloc
from memory_profiler import memory_usage
import sys
sys.path.append("../..")
import torch 
from torch_fem.sparse.solve.torch_solve import bicgstab, cg

class SimpleMemoryHook(cp.cuda.MemoryHook):
    def __init__(self):
        self.alloc_size = 0
        self.free_size = 0
        self.cur_size = 0
        self.peak_size = 0

    def malloc_preprocess(self, device_id, size, mem_size):
        self.alloc_size += size
        self.cur_size += size
        self.peak_size = max(self.peak_size, self.cur_size)

    def free_preprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        self.free_size += mem_size
        self.cur_size -= mem_size

    def print_report(self):
        print(f'Total allocated: {self.alloc_size} bytes')
        print(f'Total freed: {self.free_size} bytes')
        print(f'Current memory usage: {self.alloc_size - self.free_size} bytes')


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

def measure_time(func, *args):
    """Measure execution time of a function."""
    start_time = time.perf_counter()
    func(*args)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time

def measure_memory(func, *args):
    """Measure peak memory usage of a function."""
    def wrapper():
        func(*args)
    peak_mem = memory_usage(wrapper, max_usage=True)
    return peak_mem

def measure_cuda_memory(func, *args):
    """Measure peak memory usage of a function."""
    with SimpleMemoryHook() as hook:
        func(*args)
        cuda_peak = (hook.peak_size)/1e6
    def wrapper():
        func(*args)
    cpu_peak = memory_usage(wrapper, max_usage=True)
    return cuda_peak + cpu_peak

def measure_torch_memory(func, *args):
    """Measure peak memory usage of a function."""
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            func(*args)

    cuda_peak = (prof.key_averages().cuda_memory_usage)/1e6
    return cuda_peak

def measure_torch_cpu_memory(func, *args):
    """Measure peak memory usage of a function."""
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            func(*args)

    cpu_peak = (prof.key_averages().cpu_memory_usage)/1e6
    return cpu_peak
    
def solve_scipy(A, b):
    """ Solve using Scipy """
    A = A.tocsc()
    x = scipy.sparse.linalg.splu(A).solve(b)
    
def solve_petsc(A, b):
    """ Solve using PETSc """
    # Convert to PETSc matrix
    A = A.tocsr()
    A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    b_petsc = PETSc.Vec().createWithArray(b)

    # Solver setup
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()
    # Select the BiCGSTAB solver
    ksp.setType('bcgs')

    # (Optional) Set a preconditioner, e.g., ILU
    pc = ksp.getPC()
    pc.setType('ilu')
    # Solve
    x_petsc = b_petsc.duplicate()
    ksp.solve(b_petsc, x_petsc)

def solve_petsc_cusparse(A, b):
    """ Solve using PETSc """
    A = A.tocsr()
    A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    A_petsc.setType('aijcusparse')
    b_petsc = PETSc.Vec().createWithArray(b)

    # Solver setup
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()

    # Solve
    x_petsc = b_petsc.duplicate()
    ksp.solve(b_petsc, x_petsc)

def solve_cupy(A, b):
    """ Solve using CuPy """
    x_cuda = cupyx.scipy.sparse.linalg.spsolve(A, b)

def solve_torch(A, b):
    """ Solve using torch """
    x = cg(A, b)

n_times = 3
size = [100, 1000, 2000, 4000, 6000]

data = {
    'size': [],
    'sample':[],
    'time': [],
    'backend': [],
    'memory': [],
}


pbar = tqdm(total=len(size)*n_times*5)
for i in size:
    A, b = create_sparse_system(i)
    A_cuda = cupyx.scipy.sparse.coo_matrix(A.tocsr())
    b_cuda = cp.array(b)
    A_th   = torch.sparse_csr_tensor(torch.from_numpy(A.tocsr().data), torch.from_numpy(A.tocsr().indices), torch.Size(A.tocsr().shape))
    b_th   = torch.from_numpy(b)
    A_th_cuda = A_th.cuda()
    b_th_cuda = b_th.cuda()
    for backend, fn, params in zip(
        ["Scipy", "PETSc", "CuPy", "Torch", "Torch-CUDA"],
        [solve_scipy, solve_petsc, solve_cupy, solve_torch, solve_torch_cuda],
        [(A, b), (A, b), (A_cuda, b_cuda), (A_th, b_th), (A_th_cuda, b_th_cuda)]
        ):
            for  j in range(n_times):
                data['size'].append(i)
                data['sample'].append(j)
                data['time'].append(measure_time(fn, *params))
                data['backend'].append(backend)
                if backend == "CuPy":
                    data['memory'].append(measure_cuda_memory(fn, *params))
                elif backend == "Torch-CUDA":
                    data['memory'].append(measure_torch_memory(fn, *params))
               elif backend == "Torch":
                    data['memory'].append(measure_torch_cpu_memory(fn, *params))
                else:
                    data['memory'].append(measure_memory(fn, *params))
                pbar.update(1)
                pbar.set_postfix({
                    "size":i,
                    "sample":j,
                    "backend":backend,
                    "time":f"{data['time'][-1]:7.5f}s",
                    "memory":f"{data['memory'][-1]:7.5f}MB"
                })

data["time in s"] = data.pop("time")
data["memory in MB"] = data.pop("memory")

df = pd.DataFrame(data)

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

sns.lineplot(x='size', y='time in s', hue="backend",  data=df, ax=ax[0])
sns.lineplot(x='size', y='memory in MB', hue="backend", data=df, ax=ax[1])
for i in range(2):
    ax[i].set_yscale("log")
    ax[i].set_xscale("log")


plt.show()

fig.savefig("solve_compare.png")


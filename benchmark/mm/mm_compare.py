
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
from tqdm import tqdm
import tracemalloc

def create_sparse_system(n):
    """ Create a sparse linear system for testing. """
    A = scipy.sparse.random(n, n, density=0.0001, format='coo')
    b = np.ones(n)
    return A, b

def measure_time(func, *args):
    """Measure execution time of a function."""
    start_time = time.time()
    func(*args)
    elapsed_time = time.time() - start_time
    return elapsed_time

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

def measure_memory(func, *args):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak

def measure_cuda_memory(func, *args):
    """Measure peak memory usage of a function."""
    with SimpleMemoryHook() as hook:
        func(*args)
        peak = hook.alloc_size - hook.free_size
    return peak

def mm_scipy(A, b):
    """ Solve using Scipy """
    x = A @ b 
    
def mm_petsc(A, b):
    A = A.tocsr()
    """ Solve using PETSc """
    # Convert to PETSc matrix
    A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))

    # Convert 'B_np' to a PETSc vector
    B = PETSc.Vec().createSeq(len(b))
    B.setArray(b)

    # Perform matrix-vector multiplication
    C = A * B

def mm_cupy(A, b):
    """ Solve using CuPy """
    A_cuda = A.tocsr()
    b_cuda = cp.array(b)
    x_cuda = A_cuda @ b_cuda


n_times = 3
size = [100, 1000, 10000, 15000]

data = {
    'size': [],
    'sample':[],
    'time': [],
    'backend': [],
    'memory': [],
}


for i in tqdm(size):
    for  j in range(n_times):
        A, b = create_sparse_system(i)
        A_cuda = cupyx.scipy.sparse.coo_matrix(A)
        b_cuda = cp.array(b)
        data['size'].append(i)
        data['sample'].append(j)
        data['time'].append(measure_time(mm_scipy, A, b))
        data['backend'].append('Scipy')
        data['memory'].append(measure_memory(mm_scipy, A, b))
        data['size'].append(i)
        data['sample'].append(j)
        data['time'].append(measure_time(mm_petsc, A, b))
        data['backend'].append('PETSc')
        data['memory'].append(measure_memory(mm_petsc, A, b))
        data['size'].append(i)
        data['sample'].append(j)
        data['time'].append(measure_time(mm_cupy, A_cuda, b_cuda))
        data['backend'].append('CuPy')
        data['memory'].append(measure_cuda_memory(mm_cupy, A_cuda, b_cuda))


df = pd.DataFrame(data)

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')

sns.lineplot(x='size', y='time', hue="backend", data=df, ax=ax[0])
sns.lineplot(x='size', y='memory', hue="backend", data=df, ax=ax[1])

plt.show()

fig.savefig("mm_compare.png")
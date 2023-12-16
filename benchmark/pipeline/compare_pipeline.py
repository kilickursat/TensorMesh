import sys 
import os
import torch
sys.path.append("../..")
import torch_fem as thfem
import skfem
import skfem.helpers
import pandas as pd
import time
import cupy as cp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import fenics
from memory_profiler import memory_usage, profile


class SkFEM:
    def __init__(self, mesh, element="tri"):
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")

        self.mesh  = mesh
        self.element = element

    def __call__(self):
        element = skfem.ElementTriP1() if self.element == "tri" else skfem.ElementTetP1()
        @skfem.BilinearForm
        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v)) 
        
        @skfem.LinearForm
        def load(v, w):
            return 1.0*v
        basis = skfem.Basis(self.mesh, element)
        K     = skfem.asm(laplace, basis)

        f     = skfem.asm(load, basis)
        u     = skfem.solve(K,f)
        return  u 
    
class ThFEM:
    def __init__(self, mesh, batch_size=None):
        self.mesh = mesh
        self.batch_size = batch_size
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)

    def __call__(self):
        K     = self.K_asm(self.mesh.points, batch_size=self.batch_size)
        f     = self.f_asm(self.mesh.points, batch_size=self.batch_size)
       
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        u     = K.solve(f, backend=backend)
        return u 

class feFEM:
    def __init__(self, mesh):
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        self.fenics_mesh = fenics_mesh

    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        # Use fenics, not fem, for the methods inner, grad, and dx
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = fenics.inner(fenics.Constant(1), v) * fenics.dx
        u = fenics.Function(a.arguments()[0].function_space())

        fenics.solve(a == L, u)
        return u

       
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


def cpu_mem_peak(fun):
    mem = memory_usage(fun, interval=0.001, max_usage=True, include_children=True) 
    # base = memory_usage(lambda : None,interval=0.001, max_usage=True, include_children=True)
    return mem
def gpu_mem_peak(fun):
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats() 
    with SimpleMemoryHook() as hook:
        fun()
    cupy_peak = hook.peak_size
    torch.cuda.synchronize()
    torch_peak = torch.cuda.max_memory_allocated()  -start
    gpu_peak_mem = (torch_peak+ cupy_peak)/ 1e6
    return gpu_peak_mem
def timeit(fun):
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    start = time.perf_counter()
    fun()
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end-start
def plot_comparison(element_type, chara_lengths, n_times, csv_path):
    data = {
        "dofs":[],
        "backend":[],
        "time":[],
        "CPU mem in GB":[],
        "GPU mem in GB":[],
    }
    pbar = tqdm(total=len(chara_lengths)*n_times*4)

    for chara_length in chara_lengths:
        if element_type == "tri":
            mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
        elif element_type == "tetra":
            mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
        else:
            raise NotImplementedError(f"element_type={element_type} is not supported")
        # th_fem_cpu = ThFEM(mesh.clone())
        th_fem_cpu_1 = ThFEM(mesh.clone(), batch_size=1)
        # th_fem_gpu   = ThFEM(mesh.clone().to("cuda:0"))
        th_fem_gpu_1 = ThFEM(mesh.clone().to("cuda:0"), batch_size=1)
        sk_fem       = SkFEM(mesh, element=element_type)
        fe_fem       = feFEM(mesh)
        for _ in range(n_times):
            for name, fem in zip([
                                        # "torch_fem cpu(None)", 
                                        "torch_fem cpu",
                                        # "torch_fem cuda(None)", 
                                        "torch_fem cuda",
                                        "scikit-fem", 
                                        "fenics"], [
                                                    # th_fem_cpu, 
                                                    th_fem_cpu_1,
                                                    # th_fem_gpu, 
                                                    th_fem_gpu_1,
                                                    sk_fem, 
                                                    fe_fem]):

                data["dofs"].append(mesh.points.shape[0])
                data["backend"].append(name)
                data["time"].append(timeit(fem))
                data["CPU mem in GB"].append(cpu_mem_peak(fem) / 1e3)
                data["GPU mem in GB"].append(gpu_mem_peak(fem) / 1e3)
                pbar.update(1)
                pbar.set_postfix({
                    "dofs":mesh.points.shape[0],
                    "chara_length":chara_length,
                    "backend":name,
                })
    
    
    # df = {"metric":[],
    #       "value":[],
    #       "backend":[],
    #       "chara_length":[]}
    # for i in range(len(data["chara_length"])):
    #     for metric in ["time", "CPU RAM in MB", "GPU RAM in MB"]:
    #         df['metric'].append(metric)
    #         df['value'].append(data[metric][i])
    #         df['backend'].append(data["backend"][i])
    #         df['chara_length'].append(data["chara_length"][i])

    df = pd.DataFrame(data)
    df.to_csv(csv_path)
   
    fig, ax = plt.subplots(ncols=3, figsize=(15, 4))

    markers = ["o", "s", "p", "^"]
    linestyles = ["--", "--", "--", "--"]
    sns.pointplot(x="dofs", y="time", hue="backend", markers=markers, linestyles=linestyles,  data=df, ax=ax[0])
    sns.pointplot(x="dofs", y="CPU mem in GB", hue="backend",markers=markers, linestyles=linestyles,  data=df, ax=ax[1])
    sns.pointplot(x="dofs", y="GPU mem in GB", hue="backend",markers=markers, linestyles=linestyles, data=df, ax=ax[2])
    # for i in range(3):
    #     ax[i].set_xscale("log")
    #     ax[i].set_yscale("log")

    fig.tight_layout()

    return fig

def test():
    # from torch.profiler import profile, record_function, ProfilerActivity
    mesh = thfem.Mesh.gen_rectangle(chara_length=0.005, element_type="tri")
    skfem_ = SkFEM(mesh)
    start = time.perf_counter()
    skfem_()
    end = time.perf_counter()
    print(f"skfem: {end-start}s")
    fefem = feFEM(mesh)
    start = time.perf_counter()
    fefem()
    end = time.perf_counter()
    print(f"fenics: {end-start}s")
    fem = ThFEM(mesh.cuda(), batch_size=1)
    start = time.perf_counter()
    fem()
    end = time.perf_counter()
    print(f"torch_fem: {end-start}s")
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              record_shapes=True, 
    #              profile_memory=True, 
    #              with_stack=True,
    #              with_modules = True
    #              ) as prof:
    #     with record_function("fem"):
    #         fem()
    # with SimpleMemoryHook() as hook:
    #     fem()
    # cpu_peak = cpu_mem_peak(fem)
    # print(f"cpu peak: {cpu_peak}")
    # prof.export_chrome_trace("trace.json")
    # prof.export_memory_timeline("mem.json")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=10))

    import tracemalloc

    tracemalloc.start()
    
    fem()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    peak_memory = max(stat.size for stat in top_stats) / (1024 * 1024)

    print(f"trace_malloc peak memory: {peak_memory} MB")

    @profile
    def wrapper():
        K     = fem.K_asm(fem.mesh.points, batch_size=1)
        f     = fem.f_asm(fem.mesh.points, batch_size=1)
        u     = K.solve(f, backend="cupy")
    wrapper()

    def dummy():
        pass

    cpid_mem = memory_usage(dummy, max_usage=True)
    spid_mem = memory_usage(fem, max_usage=True)
    print(f"cpu_mem_peak: {spid_mem}/{cpid_mem} MB")
    # print(f"memory_usage peak memory: {cpu_mem_peak(fem)} MB")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == '__main__':

    fig = plot_comparison(
        element_type="tri",
        chara_lengths=[0.05, 0.01, 0.005],
        n_times=5,
        csv_path="compare_2d.csv",
    )
    fig.savefig("compare_2d.png")

    fig = plot_comparison(
        element_type="tetra",
        chara_lengths=[0.1,  0.08, 0.05],
        n_times=5,
        csv_path="compare_3d.csv",
    )

    fig.savefig("compare_3d.png")

    # test()

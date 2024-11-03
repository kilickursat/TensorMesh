import sys 
import os
import torch
sys.path.append("../..")
import tensormesh as fem
import skfem
import skfem.helpers
import pandas as pd
import time
import cupy as cp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import fenics
from memory_profiler import memory_usage

class SkFEMAsmTri:
    def __init__(self, mesh):

        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v))

        self.bilinear = skfem.BilinearForm(laplace)

        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")
       
        self.basis = skfem.Basis(mesh, skfem.ElementTriP1())

        os.remove("tmp.msh")

    def __call__(self):
        return skfem.asm(self.bilinear, self.basis)
    
class SkFEMAsmTet:
    def __init__(self, mesh):

        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v))

        self.bilinear = skfem.BilinearForm(laplace)

        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")

        self.basis = skfem.Basis(mesh, skfem.ElementTetP1())

        os.remove("tmp.msh")

    def __call__(self):
        return skfem.asm(self.bilinear, self.basis)

class ThFEMAsmCPU:
    def __init__(self, mesh, batch_size=None):

        class KAsm(fem.ElementAssembler):
            def forward(self, gradu, gradv):
                return fem.dot(gradu, gradv)

        self.asm = KAsm.from_mesh(mesh, quadrature_order=2)
        self.points = mesh.points
        self.batch_size = batch_size
    def __call__(self):
        with torch.no_grad():
            return self.asm(self.points, batch_size=self.batch_size)

class ThFEMAsmCUDA:
    def __init__(self, mesh, batch_size=None):

        class KAsm(fem.ElementAssembler):
            def forward(self,gradu, gradv):
                return fem.dot(gradu, gradv)

        mesh = mesh.cuda()

        self.asm = KAsm.from_mesh(mesh, quadrature_order=2)
        self.points = mesh.points
        self.batch_size = batch_size

    def __call__(self):
        with torch.no_grad():
            result = self.asm(self.points, batch_size=self.batch_size)
            torch.cuda.synchronize()
            return result
        
class feFEMAsm:
    def __init__(self, mesh):
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        V = fenics.FunctionSpace(fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)

        # Use fenics, not fem, for the methods inner, grad, and dx
        self.a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx

    def __call__(self):
        return fenics.assemble(self.a)
       
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

    def free_preprocess(self, device_id, mem_size, mem_ptr):
        self.free_size += mem_size
        self.cur_size -= mem_size

    def print_report(self):
        print(f'Total allocated: {self.alloc_size} bytes')
        print(f'Total freed: {self.free_size} bytes')
        print(f'Current memory usage: {self.alloc_size - self.free_size} bytes')



def plot_comparison(element_type, chara_lengths, n_times, csv_path, ax_time, ax_mem):
    data = {
        "chara_length":[],
        "assembler":[],
        "time":[],
        "CPU mem in GB":[],
        "GPU mem in GB":[],
    }
    pbar = tqdm(total=len(chara_lengths)*n_times*6)

    for chara_length in chara_lengths:
        if element_type == "tri":
            mesh = fem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
        elif element_type == "tetra":
            mesh = fem.Mesh.gen_cube(chara_length=chara_length)
        else:
            raise NotImplementedError(f"element_type={element_type} is not supported")
        th_asm_cpu = ThFEMAsmCPU(mesh)
        th_asm_gpu = ThFEMAsmCUDA(mesh)
        th_asm_cpu_1 = ThFEMAsmCPU(mesh, batch_size=1)
        th_asm_gpu_1 = ThFEMAsmCUDA(mesh, batch_size=1)
        sk_asm = {
            "tri":SkFEMAsmTri,
            "tetra":SkFEMAsmTet
        }[element_type](mesh)
        fe_asm = feFEMAsm(mesh)
        for _ in range(n_times):
            for name, assembler in zip([
                                        # "torch_fem cpu(None)", 
                                        "torch_fem cpu(1)",
                                        # "torch_fem cuda(None)", 
                                        "torch_fem cuda(1)",
                                        "scikit-fem", 
                                        "fenics"], [
                                                    # th_asm_cpu, 
                                                    th_asm_cpu_1,
                                                    # th_asm_gpu, 
                                                    th_asm_gpu_1,
                                                    sk_asm, 
                                                    fe_asm]):
                start = time.time()
                assembler()
                end = time.time()
                if "cuda" in name:
                    torch.cuda.reset_peak_memory_stats()
                    # with SimpleMemoryHook() as hook:
                    assembler()
                    # peak = hook.alloc_size - hook.free_size
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated() / 1e6
                    # peak_mem = max(peak, peak_mem)
                else:
                cpu_peak_mem = memory_usage(assembler, max_usage=True)
                gpu_pea
                data["chara_length"].append(chara_length)
                data["assembler"].append(name)
                data["time"].append(end-start)
                data["memory in MB"].append(peak_mem)
                pbar.update(1)
                pbar.set_postfix({
                    "chara_length":chara_length,
                    "assembler":name,
                    "time":f"{end-start:7.5f}s"
                })
    df = pd.DataFrame(data)
    df.to_csv(csv_path)
    markers = ["o", "s", "p", "^"]
    linestyles = ["--", "--", "--", "--"]
    sns.pointplot(x="chara_length", y="time",
             hue="assembler",data=df,ax=ax_time, markers=markers, linestyles=linestyles)
    sns.pointplot(x="chara_length", y="memory in MB",
                hue="assembler",data=df,ax=ax_mem, markers=markers, linestyles=linestyles)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")

if __name__ == '__main__':

    fig, ax = plt.subplots(ncols=2, figsize=(12,  6))
    plot_comparison(
        element_type="tri",
        chara_lengths=[0.05, 0.01, 0.005, 0.002],
        n_times=5,
        csv_path="compare_2d.csv",
        ax_time = ax[0],
        ax_mem   = ax[1]
    )

    fig.savefig("compare_2d.png")

    fig, ax = plt.subplots(ncols=2, figsize=(12,  6))

    plot_comparison(
        element_type="tetra",
        chara_lengths=[0.1,  0.05, 0.02],
        n_times=5,
        csv_path="compare_3d.csv",
        ax_time = ax[0],
        ax_mem  = ax[1]
    )

    fig.savefig("compare_3d.png")

    




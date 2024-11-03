import sys 
import os
import torch
sys.path.append("../..")
import tensormesh as thfem
import skfem
import skfem.helpers
import pandas as pd
import time
import cupy as cp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import fenics
import jax 
import jax_fem
import jax.numpy as jnp
from tensormesh.profile import TimeProfiler, CPUProfiler, CUDAProfiler, get_max_memory_for_index, get_memory_for_index
import argparse
import numpy as np
import gc

class SkFEM:
    def __init__(self, mesh):

        if mesh.default_element_type.startswith("tri"):
            element = "tri"
        else:
            element = "tetra"
        breakpoint()
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        skfem_mesh = skfem.Mesh.load("tmp.msh")

        self.mesh  = skfem_mesh
        self.element = element
        self.boundary = mesh.boundary_mask.numpy()

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
        u     = skfem.solve(*skfem.condense(K,f, D=self.boundary))
        return  u 
    
class TmFEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)
        self.condenser = thfem.Condenser(self.mesh.boundary_mask)

    def __call__(self):
        K     = self.K_asm(self.mesh.points)
        f     = self.f_asm(self.mesh.points)
        K_, f_ = self.condenser(K, f)
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "torch"
        u_    = K_.solve(f_, backend=backend)
        u     = self.condenser.recover(u_)
        return u 

class feFEM:
    def __init__(self, mesh):
        self.deivce = mesh.device
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        self.fenics_mesh = fenics_mesh

        boundary_facets = fenics.MeshFunction('size_t', fenics_mesh, fenics_mesh.topology().dim()-1)
        boundary_facets.set_all(0)

        # Mark facets as boundary if any of its nodes are in the boundary mask
        for facet in fenics.SubsetIterator(boundary_facets, 0):
            for vertex in facet.entities(0):  # Checking the vertices of each facet
                if mesh.boundary_mask[vertex]:
                    boundary_facets[facet] = 1
                    break
        
        self.boundary_facets = boundary_facets
        


    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        bc = fenics.DirichletBC(V, fenics.Constant(0), self.boundary_facets, 1)
        # Use fenics, not fem, for the methods inner, grad, and dx
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = fenics.inner(fenics.Constant(1), v) * fenics.dx
        u = fenics.Function(a.arguments()[0].function_space())

        if self.deivce.type == "cuda":
            # Use GPU solver if available
            solver_params = {
                "linear_solver": "bicgstab",
                "preconditioner": "ilu",
                "krylov_solver": {"absolute_tolerance": 1e-10},
                "petsc_ksp_cuda": True,
                "petsc_pc_cuda": True
            }
        else:
            # Use CPU solver
            solver_params = {
                "linear_solver": "bicgstab", 
                "preconditioner": "ilu"
            }
            
        fenics.solve(a == L, u, bc, solver_parameters=solver_params)
        return u

class jaxFEM:
    def __init__(self, mesh):
        self.device = mesh.device
        mesh.save("tmp.msh", file_format='gmsh')
        self.mesh = jax_fem.generate_mesh("tmp.msh")
        self.boundary_mask = mesh.boundary_mask.numpy()
    def __call__(self):
        class PoissonProblem(jax_fem.Problem):
            def get_tensor_map(self):
                return lambda x: x  # Identity map for Laplace operator

            def get_mass_map(self):
                def mass_map(u, x):
                    # Constant source term of 1.0
                    return jnp.ones((1,))
                return mass_map

        # Create problem instance
        problem = PoissonProblem(
            mesh=self.mesh, 
            vec=1,  # scalar problem
            dim=self.mesh.dim,  # dimension from mesh
            dirichlet_bc_info=[
                [lambda x: self.boundary_mask],  # boundary condition locations
                [0],  # component to apply BC
                [lambda x: 0.0]  # BC value
            ]
        )

        # Solve and return solution
        if self.device == 'cuda':
            with jax.default_device(jax.devices('gpu')[0]):
                sol = jax_fem.solver(problem)
        else:
            with jax.default_device(jax.devices('cpu')[0]):
                sol = jax_fem.solver(problem)
                
        # Convert solution to torch tensor and move to correct device
        return torch.from_numpy(sol[0]).to(self.device)

def benchmark_fem(element_type, chara_length, backend=TmFEM, device="cuda", ntimes:int=3, warmup:int=2):
    """Run benchmarks for different FEM implementations"""
    # Create mesh based on element type
    if element_type == "tri":
        mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
    elif element_type == "tetra":
        mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
    else:
        raise NotImplementedError(f"element_type={element_type} is not supported")


    mesh = mesh.to(device)
    fem = lambda: backend(mesh)

    # Warmup run
    for _ in range(warmup):
        fem()
    
    # Initialize metrics storage
    metrics = {
        'gpu_peak': [],
        'gpu_mean': [], 
        'cpu_peak': [],
        'cpu_mean': [],
        'times': []
    }

    # Run benchmarks
    for _ in range(ntimes):
        # CPU profiling
        with CPUProfiler() as cpu_prof:
            fem()
        metrics['cpu_peak'].append(cpu_prof.max())
        metrics['cpu_mean'].append(cpu_prof.mean())
        
        # GPU profiling if applicable
        if mesh.device.type == "cuda":
            with CUDAProfiler(mesh.device.index) as cuda_prof:
                fem()
            metrics['gpu_peak'].append(cuda_prof.max())
            metrics['gpu_mean'].append(cuda_prof.mean())
        else:
            metrics['gpu_peak'].append(0)
            metrics['gpu_mean'].append(0)

        # Time profiling
        with TimeProfiler(only_cpu=device_index < 0) as time_prof:
            fem()
        metrics['times'].append(time_prof.time)
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({
            "dofs": (~mesh.boundary_mask).sum().item(),
            "chara_length": chara_length,
            "backend": target,
        })

    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    # Create mask for valid measurements
    valid_mask = np.ones_like(metrics['times'], dtype=bool)
    
    # Update data dictionary with results
    n_valid = valid_mask.sum()
    data_updates = {
        "CPU peak mem in MB": metrics['cpu_peak'][valid_mask].tolist(),
        "CPU mean mem in MB": metrics['cpu_mean'][valid_mask].tolist(),
        "GPU peak mem in MB": metrics['gpu_peak'][valid_mask].tolist(),
        "GPU mean mem in MB": metrics['gpu_mean'][valid_mask].tolist(),
        "time in s": metrics['times'][valid_mask].tolist(),
        "chara length": [chara_length] * n_valid,
        "degree of freedom": [mesh.n_points] * n_valid,
        "backend": [target] * n_valid
    }
    
    for key, values in data_updates.items():
        data[key].extend(values)


def draw_error_bar(data, x, y, hue, ax):
    """Draw error bars with consistent styling"""
    styles = {
        'line_styles': ["-", "--", "-.", ":"],
        'markers': ["o", "s", "p", "^"],
        'colors': ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    }
    
    for i, (group_name, group_data) in enumerate(data.groupby(hue)):
        xs, means, stds = [], [], []
        
        # Calculate statistics for each group
        for x_val, subgroup in group_data.groupby(x):
            xs.append(x_val)
            means.append(subgroup[y].mean())
            stds.append(subgroup[y].std())
            
        # Plot error bars
        ax.errorbar(
            xs, means, 
            yerr=np.array(stds)*3,
            color=styles['colors'][i],
            marker=styles['markers'][i],
            linestyle=styles['line_styles'][i],
            capsize=4.0,
            alpha=0.5,
            label=group_name
        )
    
    # Configure axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim(left=min(xs)/2, right=max(xs)*2)
    ax.legend()


def plot_comparison( 
                    config:str,
                    element_type:str,
                    n_times:int, 
                    force:bool = False):

    df = {
        "degree of freedom": [],
        "backend": [],
        "time in s": [],
        "chara length": [],
        "CPU peak mem in MB": [],
        "CPU mean mem in MB": [],
        "GPU peak mem in MB": [],
        "GPU mean mem in MB": [],
    }

    element2dimension = {
        "tri":"2d",
        "tetra":"3d"
    }

    # Setup progress bar
    total = sum([len(conf[f"{element2dimension[element_type]}_lengths"]) for conf in config.values()])
    pbar = tqdm(total = total)
    for key, conf in config.items():
        
    
        for chara_length in conf[f"{element2dimension[element_type]}_lengths"]:
        
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            
            # Run benchmark
            time,memory = benchmark_fem(element_type, chara_length, conf['backend'], conf['device'], n_times)

            pbar.update(1)
            pbar.set_postfix({
                "backend":conf['backend'],
                "device":conf["device"],
                "time":f"{time.mean():.3g}({time.std():.3g})s",
                "memory":f"{memory:.3g}MB"
                })
    
    
    # Create plots
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    metrics = [
        ("time in s", "Time"),
        ("CPU peak mem in MB", "CPU Memory"),
        ("GPU peak mem in MB", "GPU Memory")
    ]
    
    for ax, (metric, _) in zip(axes, metrics):
        draw_error_bar(pd.DataFrame.from_dict(df), 
                      "degree of freedom", metric, "backend", ax)
    
    fig.tight_layout()
    return fig


def test():
    """Run performance tests and memory profiling"""
    mesh = thfem.Mesh.gen_rectangle(chara_length=0.005, element_type="tri")
    
    # Test different implementations
    implementations = {
        'skfem': SkFEM(mesh),
        'fenics': feFEM(mesh),
        'torch_fem': ThFEM(mesh.cuda(), batch_size=1)
    }
    
    for name, impl in implementations.items():
        start = time.perf_counter()
        impl()
        duration = time.perf_counter() - start
        print(f"{name}: {duration:.3f}s")

    # Memory tracking with tracemalloc
    tracemalloc.start()
    implementations['torch_fem']()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    peak_memory = max(stat.size for stat in top_stats) / (1024 * 1024)
    print(f"\nTrace_malloc peak memory: {peak_memory:.2f} MB")
    
    print("\n[ Top 10 Memory Allocations ]")
    for stat in top_stats[:10]:
        print(stat)

    # Memory profiling
    @profile
    def profile_solve():
        fem = implementations['torch_fem']
        K = fem.K_asm(fem.mesh.points, batch_size=1)
        f = fem.f_asm(fem.mesh.points, batch_size=1)
        u = K.solve(f, backend="cupy")
    
    profile_solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_index", type=int, default=-1)
    parser.add_argument("-n", "--num_dofs", type=int, default=3)
    parser.add_argument("-t", "--times", type=int, default=5)
    parser.add_argument("--only_2d", action="store_true")
    parser.add_argument("--only_3d", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--server", action="store_true", help="run on server")
    args = parser.parse_args()

    mem = get_max_memory_for_index(args.device_index) - get_memory_for_index(args.device_index)

    # Configuration for different backends
    config = {
        "tensormesh_cpu": {
            "backend" : TmFEM,
            "device"  : "cpu",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001, 0.0005],
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015, 0.01]
        },
        "tensormesh_cuda": {
            "backend" : TmFEM,
            "device"  : "cuda",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.0012] + 
                         ([0.0005, 0.00025] if args.server else []),
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015] + 
                         ([0.01, 0.008] if args.server else [])
        },
        "fenics_cpu": {
            "backend": feFEM,
            "device"  : "cpu",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001, 0.0005],
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015, 0.01]
        },
        "fenics_cuda": {
            "backend": feFEM,
            "device": "cuda",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001] + 
                         ([0.0005, 0.00025] if args.server else []),
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015] + 
                         ([0.01, 0.008] if args.server else [])
        },
        "skfem": {
            "backend": SkFEM,
            "device": "cpu",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001],
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02]
        },
        "jax_cpu": {
            "backend": None,
            "device": "cpu",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001, 0.0005],
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015, 0.01]
        },
        "jax_cuda": {
            "backend": None,
            "device": "cuda",
            "2d_lengths": [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001] + 
                        ([0.0005, 0.00025] if args.server else []),
            "3d_lengths": [0.2, 0.1, 0.05, 0.04, 0.02, 0.015] + 
                        ([0.01, 0.008] if args.server else [])
        }

    }


    if not torch.cuda.is_available():
        config.pop("tensormesh_cuda")
        config.pop("fenics_cuda") 
        config.pop("jax_cuda")

    # Run 2D benchmarks
    if not args.only_3d:
        fig = plot_comparison(
            element_type="tri",
            config = config,
            n_times=args.times,
            csv_path="compare_linear_poisson_2d.csv",
            force=args.force,
            device_index=args.device_index
        )
        fig.savefig("compare_linear_poisson_2d.png")
        fig.savefig("compare_linear_poisson_2d.pdf")

    # Run 3D benchmarks
    if not args.only_2d:
        fig = plot_comparison(
            element_type="tetra",
            chara_lengths=config["3d_lengths"],
            backends=config["backends"],
            n_times=args.times,
            csv_path="compare_linear_poisson_3d.csv",
            force=args.force,
            device_index=args.device_index
        )
        fig.savefig("compare_linear_poisson_3d.png")
        fig.savefig("compare_linear_poisson_3d.pdf")


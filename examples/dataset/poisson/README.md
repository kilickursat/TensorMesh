# Poisson Equation Dataset Generation

Batch generation of Poisson equation solutions for machine learning workflows.

## Problem Setup

- **PDE:** $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$
- **Geometry:** Unit square (2D) or unit cube (3D)
- **Source Term:** Multi-frequency Fourier series via `PoissonMultiFrequency` / `PoissonMultiFrequency3D`
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)

## Features

- **Batch solve:** Solves all samples simultaneously using multi-RHS sparse LU (one factorization, many back-substitutions)
- **GPU acceleration:** Automatic GPU detection with CuPy backend
- **Analytical comparison:** Exact solution available from the same Fourier series
- **Benchmarking:** Sweeps batch sizes from 1 to 1024 and reports per-problem timing

## Usage

```bash
python poisson_dataset.py --mode 2d      # 2D batch
python poisson_dataset.py --mode 3d      # 3D batch
python poisson_dataset.py --mode bench   # batch-size scaling benchmark
python poisson_dataset.py --mode all     # all of the above
```

## Output

- `poisson_batch_solver_2d.png`: source term, FEM solution, analytical solution, error (sample 0)
- `poisson_batch_solver_3d.vtu`: 3D solution (open with ParaView)

# 2D Lid-Driven Cavity

Steady-state incompressible Navier-Stokes in a 2D square cavity with a moving lid.

## Problem Setup

- **Geometry:** Unit square [0, 1]^2 (triangular mesh)
- **Physics:** Incompressible Navier-Stokes, Picard linearization
- **Boundary Conditions:**
  - Top lid (y=1): u_x=1, u_y=0 (moving wall)
  - All other boundaries: no-slip (u=0)
- **Parameters:** Re=100 (default), rho=1.0, mu=1/Re
- **Stabilization:** SUPG/PSPG

## Usage

```bash
python cavity.py
```

## Output

- `cavity_results.png`: speed and pressure contour plots

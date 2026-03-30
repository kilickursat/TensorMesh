# 3D Lid-Driven Cavity

Steady-state incompressible Navier-Stokes in a 3D unit cube with a moving top face.

## Problem Setup

- **Geometry:** Unit cube [0, 1]^3 (tetrahedral mesh)
- **Physics:** Incompressible Navier-Stokes, Picard linearization (dimension-generic assembler)
- **Boundary Conditions:**
  - Top face (y=1): u_x=1, u_y=0, u_z=0 (moving lid)
  - All other faces: no-slip (u=0)
- **Parameters:** Re=100 (default), rho=1.0, mu=1/Re
- **Stabilization:** SUPG/PSPG

## Usage

```bash
python cavity_3d.py
```

## Output

- `cavity_3d.vtu`: volumetric data (open in ParaView)
- `cavity_3d.png`: cross-section slices (speed + pressure at z=0.5) via PyVista

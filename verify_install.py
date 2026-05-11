"""Verify a TensorMesh install: solve a tiny Poisson problem on CPU
(and on GPU if available), then report which sparse-solver backends
are wired up."""

import math
import platform
import sys
import time

import torch


def solve_poisson(device):
    from tensormesh import ElementAssembler, NodeAssembler, Mesh, Condenser

    mesh = Mesh.gen_rectangle(chara_length=0.1).to(device)

    class Laplace(ElementAssembler):
        def forward(self, gradu, gradv):
            return gradu @ gradv

    class Source(NodeAssembler):
        def forward(self, v, f):
            return f * v

    x, y = mesh.points[:, 0], mesh.points[:, 1]
    f_vals = 2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

    K = Laplace.from_mesh(mesh)()
    b = Source.from_mesh(mesh)(point_data={"f": f_vals})

    cond = Condenser(mesh.boundary_mask)
    K_, b_ = cond(K, b)
    u = cond.recover(K_.solve(b_))

    u_exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return float((u - u_exact).norm() / u_exact.norm())


def probe_backends():
    out = {"scipy": (True, ""), "torch": (True, "")}
    try:
        import torch_sla  # noqa: F401
        out["torch_sla"] = (True, "")
    except ImportError:
        out["torch_sla"] = (False, "pip install torch-sla")
    try:
        import pyamg  # noqa: F401
        out["amg"] = (True, "")
    except ImportError:
        out["amg"] = (False, "pip install pyamg")
    from tensormesh.sparse import is_petsc_available, is_cupy_available
    out["petsc"] = (is_petsc_available,
                    "" if is_petsc_available else "pip install petsc4py")
    out["cupy"]  = (is_cupy_available and torch.cuda.is_available(),
                    "" if is_cupy_available else "pip install cupy")
    out["cudss"] = (torch.cuda.is_available(),
                    "" if torch.cuda.is_available() else "requires CUDA + libcudss")
    return out


def main():
    import tensormesh
    print("TensorMesh smoke test")
    print("=" * 40)
    print(f"tensormesh : {tensormesh.__version__}")
    print(f"torch      : {torch.__version__}")
    try:
        import torch_sla
        print(f"torch-sla  : {getattr(torch_sla, '__version__', 'unknown')}")
    except ImportError:
        print(f"torch-sla  : MISSING")
    print(f"python     : {sys.version.split()[0]} on {platform.system().lower()}")
    print()

    t0 = time.perf_counter()
    err = solve_poisson("cpu")
    print(f"[CPU ] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter()-t0:.2f} s")

    if torch.cuda.is_available():
        t0 = time.perf_counter()
        err = solve_poisson("cuda")
        print(f"[CUDA] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter()-t0:.2f} s")
    else:
        print(f"[CUDA] not available, skipping GPU test")

    print()
    print("Sparse solver backends:")
    for name, (ok, hint) in probe_backends().items():
        mark = "OK " if ok else "-- "
        suffix = f"  ({hint})" if (not ok and hint) else ""
        print(f"  {name:<10s}: {mark}{suffix}")

    print()
    print("All required checks passed.")


if __name__ == "__main__":
    main()
# Roadmap

Planned directions for core source code of TensorMesh, roughly ordered by priority. Solver-side items (backends, complex adjoint, multi-GPU) live in the [torch-sla ROADMAP](https://github.com/sparsexlab/torch-sla).

_Last updated: 2026-05._

## 1. Mixed (multi-field / block) assembly for Lagrange spaces

Support weak forms over several Lagrange function spaces at once — e.g. Taylor–Hood P2–P1 for Stokes / incompressible flow — assembling block systems with off-diagonal coupling blocks. This needs **no new element type**: it extends the assembler's argument dispatch and the projector's scatter to a block DOF layout. Highest priority because it removes the manual block/offset bookkeeping that currently makes multi-field (fluids) assemblers far too verbose, and it is the foundation that the complex (item 2) and mixed-element (item 3) work both reuse.

## 2. Complex-valued FEM → Helmholtz, PML, metamaterial topology optimization

Unblock the assembly stack for complex-valued systems so a complex element matrix can flow end-to-end into a complex-symmetric LDLᵀ / Hermitian LDLᴴ solve — enabling time-harmonic Helmholtz with PML and, on top of it, topology optimization of acoustic and (2D / scalar) electromagnetic metamaterials.

Why it fits the current architecture: PML is a **volume** modification (complex, anisotropic coefficients `A(x)`, `c(x)` in the coordinate-stretched layer), not a boundary condition, so it maps directly onto the existing `ElementAssembler` — a tensor-valued complex coefficient is expressible via `point_data` (the assembly `einsum` ellipsis already carries tensor fields). No new facet/boundary assembler is needed.

Assembly-side work is mostly removing real-dtype assumptions; the assembly is built on complex-autograd-clean primitives (`einsum`, `index_add`):
- `ElementAssembler.type()` rejects non-`float32/float64` dtypes (geometry buffers stay real; the complex value comes from the material/coefficient via `point_data`).
- `ReduceProjector`'s `accumulate_f64` path drops the imaginary part via `.double()`.
- `SparseProjector` hardcodes a `float32` projection matrix.
- Audit `Condenser` for `.double()` / real-dtype assumptions.

Solver-side dependency: the complex solve **and the correct complex adjoint** — essential for TopOpt, where a wrong adjoint yields silently wrong design sensitivities — live in torch-sla; see item 1 of the [torch-sla ROADMAP](https://github.com/sparsexlab/torch-sla).

Topology-optimization scaffolding mostly exists: the density → SIMP → filter → OC pipeline is already proven on real problems (`tensormesh/optimizer/oc.py`, `examples/inverse_design/`). The wave objective is real (e.g. `|u|²` at a target point), so autograd's real-loss convention holds — but classic OC assumes monotone, compliance-like sensitivities, so a wave objective may want MMA instead.

Scope note: scalar complex Helmholtz (complex Lagrange) covers acoustics and 2D / scalar (TE/TM) electromagnetics. **Full-vector 3D electromagnetics** needs H(curl) Nédélec elements — gated on item 3.

## 3. P0, then Raviart–Thomas (H(div)) and Nédélec (H(curl)) elements

TensorMesh today ships **continuous Lagrange elements only** (`Line`, `Triangle`, `Quadrilateral`, `Tetrahedron`, `Hexahedron`, `Pyramid`, `Prism`, plus higher-order nodal variants), and the element abstraction is **nodal-Lagrange to the core**: scalar basis with the nodal interpolation property (`φᵢ(xⱼ) = δᵢⱼ`, see `element.py`), the standard covariant (gradient) map `∇ₓφ = J⁻ᵀ∇_ξφ`, and node-based global DOFs.

**P0 (cell-constant, discontinuous).** One DOF per cell, no continuity, no orientation — a small addition that mostly needs a "cell-DOF" carrier alongside the existing node-DOF model. P0 is the natural pressure/multiplier space, e.g. the lowest-order RT–P0 mixed Poisson / Darcy pair.

**Raviart–Thomas (H(div)), then Nédélec (H(curl)).** The substantial step — it extends three core assumptions, not a new subclass:
- **Vector-valued, non-nodal basis** — RT basis functions are vector-valued (`shape_val` gains a component dimension, `[n_q, n_basis, dim]`) and defined by facet-flux moments `∫_facet v·n = δ`, not nodal interpolation. Weak forms use the field `v` and its `div`, not a scalar gradient.
- **Piola map** — RT needs the contravariant Piola map `v = (1/det J) J v̂` to preserve normal continuity, distinct from the covariant gradient map `Transformation` implements today (no Piola exists yet).
- **Facet DOFs + orientation/sign** — global DOFs move from nodes to facets, needing a unique-facet enumeration as DOF carriers and a per-element ±1 sign convention so shared-facet normals agree. The geometric facet machinery (`get_facet`, `get_edge`, facet quadrature, `Transformation.facets`) already exists; the missing piece is the facet-DOF / orientation layer (the projector currently scatters to node indices, and `reorder` handles node permutations only).

This is comparable in scope to what scikit-fem built deliberately for H(div)/H(curl). Item 1 (block assembly) and P0 bring us to the doorstep of RT–P0; the RT/Nédélec vector-basis + Piola + facet-DOF work is the genuinely new part, and Nédélec is what unlocks full-vector 3D electromagnetics for item 2.

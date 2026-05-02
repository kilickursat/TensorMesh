Neural-Operator Dataset (Mousavi 2026)
=======================================

A complete TensorMesh-based reimplementation of the dataset
generation pipeline from

   *Imposing Boundary Conditions on Neural Operators via Learned
   Function Extensions.*
   Sepehr Mousavi, Siddhartha Mishra, Laura De Lorenzis (2026).
   `arXiv:2602.04923 <https://arxiv.org/abs/2602.04923>`_.

The original code used DOLFINx / FEniCSx; the
``examples/solid/mousavi2026/`` directory ports the entire
17-file, 18-dataset pipeline onto TensorMesh's PyTorch backend.
The payoff is GPU acceleration and end-to-end differentiability —
useful both for *generating* the training data and for the
neural-operator workflows that consume it.

This is the largest single example in the gallery; the page below
is structured as a tour rather than a code walkthrough.


The 18 datasets
---------------

Two PDE families × six geometries × three configurations (with some
combinations skipped) gives the 18 published datasets:

**Poisson (9 datasets)** — scalar PDE
:math:`-\Delta u = f` with mixed BCs:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Geometry
     - bc1
     - bc4
     - bc5
   * - Unit circle
     - all Dirichlet
     - mixed D / N / R
     - mixed + random :math:`f`
   * - :math:`[-1, 1]^2` square
     - all Dirichlet
     - mixed D / N / R
     - mixed + random :math:`f`
   * - Boomerang
     - all Dirichlet
     - mixed D / N / R
     - mixed + random :math:`f`

* **bc1**: one segment covering the whole boundary, all
  Dirichlet, with random sinusoidal data; source :math:`f` is a
  fixed centered radial cosine.
* **bc4**: four segments with random "joints" along the
  parametric boundary; each segment independently chosen
  Dirichlet / Neumann / Robin (~33 % each); non-Dirichlet
  fraction constrained to :math:`[0.2, 0.51]`.
* **bc5**: same boundary structure as bc4, but :math:`f` is also
  randomized as a sum of radial sines.

**Elasticity (9 datasets)** — 2D Neo-Hookean hyperelasticity on
hollow geometries × three materials:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Geometry
     - m1 (Steel)
     - m2 (Bone)
     - m3 (Rubber)
   * - CircleHollow
     - 2 load steps
     - 4 load steps
     - 25 load steps
   * - SquareHollow
     - 2 load steps
     - 4 load steps
     - 25 load steps
   * - BoomCircleTri
     - 2 load steps
     - 4 load steps
     - 25 load steps

The number of load steps reflects the material softness — Rubber
needs many small increments because the deformations per step are
much larger.


Pipeline at a glance
--------------------

The driver script ``generate_dataset.py`` glues together the
dataset's worth of work for one sample:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │                       generate_dataset.py                     │
   │            (CLI entry point + sample loop, ~400 lines)        │
   └─────────┬──────────┬──────────┬──────────┬──────────────────┘
             ▼          ▼          ▼          ▼
        mesh_gen.py  boundary.py  bc_segments.py  geometry.py
       (Gmsh-backed  (random BC   (angle-based     (boomerang,
        meshing)      generators)  segment maps)    SmoothJoint)
             │          │          │
             ▼          ▼          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │           poisson_solver.py     elasticity_solver.py          │
   │          (LaplaceAssembler +    (NeoHookean + LBFGS,           │
   │           Robin/Neumann BCs)     incremental loading)         │
   └──────────────────────────────────────────────────────────────┘
             │
             ▼
        sdf.py    harmonic_extension.py    output.py
       (signed    (smooth boundary-data    (HDF5 with
        dist.)     interior extensions)     vlen arrays)
             │
             ▼
                       data/<dataset>.nc

For each sample the driver:

1. Generates the mesh once per geometry (cached).
2. Pre-computes the signed distance field at mesh nodes and on a
   :math:`256 \times 256` regular grid.
3. Draws random BCs subject to the validity constraints.
4. Assigns boundary nodes to segments by parametric angle, builds
   per-component Dirichlet / Neumann / Robin masks, evaluates BC
   functions at nodes.
5. Solves the PDE.
6. Computes harmonic extensions of the BC data
   (:math:`\alpha, \beta, g`) into the interior — the neural
   operator consumes these as input features.
7. Writes coordinates, solution, derived fields, SDF, and BC data
   into the variable-length HDF5 schema.

Roughly 100 lines of TensorMesh code do the actual FEM work; the
other ~1500 lines are bookkeeping (random BC generation, segment
assignment, validity checks, HDF5 packing) — the dataset machinery
that the original FEniCSx code spends a similar fraction of its
budget on.


Where TensorMesh enters
-----------------------

* **Mesh generation** — :class:`~tensormesh.MeshGen` for the
  simple geometries; raw Gmsh API for boomerang and the hollow
  geometries with custom hole curves.
* **Poisson solver** —
  :class:`~tensormesh.LaplaceElementAssembler` for the stiffness;
  a custom :class:`~tensormesh.NodeAssembler` for the source RHS;
  edge-based integration via
  :class:`~tensormesh.FacetAssembler` for Neumann and Robin
  contributions; :class:`~tensormesh.Condenser` for static
  condensation of the Dirichlet DOFs.
* **Elasticity solver** — a custom Neo-Hookean assembler (same
  energy density as :doc:`hyperelastic_beam`), L-BFGS energy
  minimization with incremental loading, component-wise BCs.
* **Harmonic extension** — three extra Laplace solves per BC
  dimension to extend boundary data smoothly into the domain.

The fact that all of these speak the same TensorMesh API is what
makes the script readable in spite of its scope.


Output format (HDF5)
--------------------

Each dataset is one ``.nc`` file. Sample fields per record:

.. code-block:: text

   coordinates   [N, 1, 2]      var-length: mesh node coordinates
   bbox/grid     [N, 1, 2, 256] x/y of the regular SDF grid
   bbox/sdf      [N, 1, 1, 256, 256] SDF on the grid
   interior/
     sdf          [N, 1, 1]    var-length: SDF at mesh nodes
     sdf_grad     [N, 1, 2]    var-length: SDF gradient at nodes
     solution     [N, 1, ndims]  var-length: FEM solution
     source       [N, 1, 1]    var-length (Poisson only)
     strain       [N, 1, 4]    var-length (Elasticity only)
     cauchystress [N, 1, 4]    var-length (Elasticity only)
     extensions/{0,1}/{alpha,beta,g}  harmonic extensions
   boundaries/{0,1}/{dirichlet,neumann,robin}/
     indices       boundary node indices
     g             prescribed values
     alpha         Robin coefficient (Robin only)

Variable-length arrays are stored via ``h5py.vlen_dtype`` so that
samples with different mesh node counts can coexist in one file.


Running it
----------

A single dataset:

.. code-block:: bash

   cd examples/solid/mousavi2026

   # 10 Poisson samples on the unit circle, all-Dirichlet
   python generate_dataset.py --problem poisson  --shape circle \
                              --id bc1 --size 10 --save-hdf5

   # 5 elasticity samples on the hollow circle, Steel
   python generate_dataset.py --problem elasticity --shape circlehollow \
                              --id m1 --size 5 --save-hdf5

All 18 datasets at once:

.. code-block:: bash

   bash batch_generate.sh 256 ./data        # 256 samples each
   bash batch_generate.sh 8448 ./data       # full published dataset

Visualize one sample per dataset for sanity checking:

.. code-block:: bash

   python visualize_all.py --output-dir ./figures


Benchmarks
----------

The directory ships ``benchmark_compare.py``,
``benchmark_tensormesh.py``, and ``benchmark_fenicsx.py`` (the last
one optional, only for users with FEniCSx installed). Their job
is to time TensorMesh's solver against FEniCSx on identical
configurations. Headline result: with a single GPU, TensorMesh's
Poisson solver is competitive with single-process FEniCSx on
small problems and several × faster on large ones, dominated by
the batched assembly (see :doc:`../dataset` for the
multi-RHS pattern that powers the same speedup in time-stepping
workloads).


What's next
-----------

* :doc:`hyperelastic_beam` — the pedagogical version of the
  Neo-Hookean energy used here.
* :doc:`emmentaler` — the same Neo-Hookean model on a 3D
  geometry with a phase-field stage.
* :doc:`../dataset` — batched ML data generation patterns that
  use the same ``[n_dof, n_batch]`` solver routing as this one.
* :doc:`../../user_guide/differentiability` — how the autograd-
  aware solve makes "FEM-in-the-loop" training tractable.

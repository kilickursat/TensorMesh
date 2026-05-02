Fluid Mechanics
===============

Six worked Navier-Stokes examples in ``examples/fluid/``, from the
2D lid-driven cavity benchmark to a Rayleigh-Bénard convection
problem coupling momentum and energy. All six share the same
underlying recipe — a custom ``NavierStokesAssembler`` (or its
transient cousin) defined in the example file, SUPG/PSPG
stabilization for equal-order P1-P1 elements, Picard linearization
for steady solves, implicit Euler for transient ones — so once you
have read :doc:`cavity` the rest are essentially geometry and
boundary-condition variations.

The ``NavierStokesAssembler`` itself lives **in the example
folder**, not in ``tensormesh.assemble``. It is intentionally
example-grade: the production-quality version will move into the
core library when the assembler API for vector-valued problems
stabilizes.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: 2D Lid-Driven Cavity
      :link: cavity
      :link-type: doc

      Steady NS at Re=100, SUPG/PSPG stabilization, Picard iteration.

   .. grid-item-card:: 3D Lid-Driven Cavity
      :link: cavity_3d
      :link-type: doc

      Same physics on a tetrahedral cube — the assembler is dimension-generic.

   .. grid-item-card:: Cylinder Flow (Vortex Shedding)
      :link: cylinder_flow
      :link-type: doc

      Transient DFG benchmark, implicit Euler, drag/lift/Strouhal post-processing.

   .. grid-item-card:: Flow Past Multiple Obstacles
      :link: flow_obstacles
      :link-type: doc

      Steady channel flow at Re=150 around six circular obstacles via MeshGen CSG.

   .. grid-item-card:: Rayleigh-Bénard Convection
      :link: rayleigh_benard
      :link-type: doc

      Boussinesq-coupled momentum + heat transport, buoyancy-driven flow.

   .. grid-item-card:: Taylor-Green Vortex
      :link: taylor_green
      :link-type: doc

      Decaying vortex with exact solution — the convergence-study showcase.


.. toctree::
   :hidden:
   :maxdepth: 1

   cavity
   cavity_3d
   cylinder_flow
   flow_obstacles
   rayleigh_benard
   taylor_green

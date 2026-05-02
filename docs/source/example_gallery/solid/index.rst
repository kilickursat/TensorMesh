Solid Mechanics
===============

Seven worked solid-mechanics examples in ``examples/solid/``,
walking from textbook linear elasticity all the way through
hyperelasticity, contact, plasticity, and a full paper-grade
dataset suite. The examples cover the three solver patterns
TensorMesh supports for solid problems:

* **Direct linear solve** for small-strain linear elasticity
  (``cantilever_beam``).
* **LBFGS energy minimization** for nonlinear problems where the
  potential energy is well-defined: hyperelasticity, contact,
  plasticity (``hyperelastic_beam``, ``hertzian_contact``,
  ``plasticity_*``, ``emmentaler``).
* **Multi-phase / staggered solvers** for problems where the
  energy depends on auxiliary fields (``emmentaler`` phase-field,
  ``mousavi2026`` dataset suite).

The order below loosely mirrors solver complexity.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Cantilever Beam
      :link: cantilever_beam
      :link-type: doc

      Linear elasticity, steel cantilever with a tip load — the simplest end-to-end recipe.

   .. grid-item-card:: Emmentaler Block
      :link: emmentaler
      :link-type: doc

      3D block with 23 holes, three progressive physics: linear, Neo-Hookean, phase-field fracture.

   .. grid-item-card:: Hyperelastic Beam
      :link: hyperelastic_beam
      :link-type: doc

      Rubber beam under torsion, compressible Neo-Hookean, LBFGS load stepping.

   .. grid-item-card:: Hertzian Contact
      :link: hertzian_contact
      :link-type: doc

      Penalty contact between a circular indenter and an elastic block.

   .. grid-item-card:: Plasticity Strip
      :link: plasticity_strip
      :link-type: doc

      2D plane-strain J2 plasticity, isotropic hardening, load / unload cycle.

   .. grid-item-card:: Plasticity 3D
      :link: plasticity_3d
      :link-type: doc

      The same J2 model on a 3D cube at 40% strain.

   .. grid-item-card:: Neural-Operator Dataset
      :link: mousavi2026
      :link-type: doc

      Full reimplementation of the 18-dataset Mousavi 2026 paper for training neural operators.


.. toctree::
   :hidden:
   :maxdepth: 1

   cantilever_beam
   emmentaler
   hyperelastic_beam
   hertzian_contact
   plasticity_strip
   plasticity_3d
   mousavi2026

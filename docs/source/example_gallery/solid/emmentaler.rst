Emmentaler Block
================

The Emmentaler example is a three-stage progression on the same
geometry — a :math:`1.0 \times 1.0 \times 1.5` block punched with
23 randomly-placed spherical holes (hence the name). The block is
clamped at the bottom and subjected to a combined tension /
bending / torsion at the top. Three scripts in
``examples/solid/emmentaler/`` solve the same boundary-value
problem with progressively richer physics:

1. **Linear elasticity** (``emmentaler_elasticity.py``) — small
   strain, single direct solve per load step.
2. **Neo-Hookean hyperelasticity** (``emmentaler_hyperelastic.py``)
   — finite strain, energy minimization.
3. **Phase-field fracture** (``emmentaler_phasefield.py``) — AT1
   damage model with staggered displacement / damage solver.

The geometry, BCs, and loading are identical across the three; the
script you run determines the constitutive model. This makes
Emmentaler a useful "scaling test" for solver complexity: the
linear-elasticity stage is a sanity check (drops out of any other
stage as the small-strain limit), and the phase-field stage is the
research-grade endpoint.


Shared setup
------------

All three scripts use the same loading: at the top face
:math:`z = T`, prescribe displacement

.. math::

   u_z &= \lambda \cdot 0.010, \\
   \kappa_\text{bend} &= \lambda \cdot 0.014, \\
   \theta_\text{tors} &= \lambda \cdot 0.048 \;\text{rad},

with :math:`\lambda \in [0, 1]` ramped over ``--load_steps``.
Material defaults are :math:`E = 12000`, :math:`\nu = 0.3`. The
mesh is generated via Gmsh with the holes as boolean cuts; pass
``--h 0.08`` for a fast smoke test, ``--h 0.03`` for a publication
mesh.


Stage 1 — linear elasticity
---------------------------

Identical to :doc:`cantilever_beam` in spirit:
:class:`~tensormesh.LinearElasticityElementAssembler`
plus a :class:`~tensormesh.Condenser` for the prescribed top
displacement, one direct solve per load step.

.. code-block:: bash

   python emmentaler_elasticity.py --h 0.08 --steps 11

Outputs include the displacement field, infinitesimal strain in
Voigt form, Cauchy stress, and von Mises equivalent — written to
a ``.vtk`` file you can open in ParaView. This is the right stage
to use as a baseline when verifying mesh resolution: under a small
:math:`\lambda` the hyperelastic and elastic solutions should
agree.


Stage 2 — Neo-Hookean hyperelasticity
-------------------------------------

The hyperelastic stage replaces the small-strain quadratic
energy with the compressible Neo-Hookean strain-energy density

.. math::

   \Psi(\mathbf{F}) \;=\;
   \frac{\mu}{2} (I_1 - 3)
   - \mu \ln J
   + \frac{\lambda}{2} (\ln J)^2,

where :math:`\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}`,
:math:`I_1 = \mathrm{tr}(\mathbf{F}^T \mathbf{F})`, and
:math:`J = \det \mathbf{F}`. The total potential
:math:`\Pi(\mathbf{u}) = \int_\Omega \Psi\,\mathrm{d}\Omega` is
minimized with PyTorch's L-BFGS optimizer, with autograd providing
the gradient. The class implementing :math:`\Psi` is the same
``NeoHookeanModel`` you see in :doc:`hyperelastic_beam`.

Incremental loading is essential: the total tension / bending /
torsion is split into ``--load_steps`` increments, and each step
warm-starts L-BFGS from the previous step's solution. Without
incrementation, L-BFGS often fails to find a basin at full load.

.. code-block:: bash

   python emmentaler_hyperelastic.py --h 0.08 --load_steps 10

Output is the same as Stage 1 but with **Green-Lagrange strain**
and **Cauchy stress** (computed from the converged deformation
gradient, not the small-strain approximations).


Stage 3 — phase-field fracture (AT1)
-------------------------------------

The phase-field stage adds a scalar damage variable
:math:`\alpha \in [0, 1]` (0 = pristine, 1 = fully cracked) to
the unknowns. The total energy is

.. math::

   \mathcal{E}(\mathbf{u}, \alpha) \;=\;
   \int_\Omega g(\alpha)\, \Psi^{+}(\mathbf{F}) \,\mathrm{d}\Omega
   \;+\;
   \int_\Omega \Psi^{-}(\mathbf{F}) \,\mathrm{d}\Omega
   \;+\;
   \frac{G_c}{c_w} \int_\Omega
   \left(\frac{w(\alpha)}{\ell} + \ell\, |\nabla \alpha|^2 \right)
   \,\mathrm{d}\Omega,

with the AT1 dissipation function :math:`w(\alpha) = \alpha`,
the Amor split into tensile / compressive parts
:math:`\Psi^{+} / \Psi^{-}`, and the regularization length scale
:math:`\ell`. The script solves it with a **staggered** alternating
minimization:

* fix :math:`\alpha`, minimize :math:`\mathcal{E}` over
  :math:`\mathbf{u}` (LBFGS),
* fix :math:`\mathbf{u}`, minimize :math:`\mathcal{E}` over
  :math:`\alpha` with the irreversibility constraint
  :math:`\alpha \geq \alpha^{n-1}` enforced by projection
  (projected LBFGS).

Repeat until both fields converge, then take the next load step.
This is the standard recipe in the phase-field-fracture literature.

.. code-block:: bash

   # Quick test (coarse, large ell for resolvability)
   python emmentaler_phasefield.py \
       --h 0.1 --ell 0.15 --Gc 0.05 --load_steps 20

   # Production parameters (h must be smaller than ell)
   python emmentaler_phasefield.py \
       --h 0.03 --ell 0.075 --Gc 0.0014 --load_steps 201

The AT1 model has a sharp elastic phase before damage initiates,
so most of the load steps are uneventful — the interesting picture
is in the last 10–20 % of the loading where cracks nucleate near
the holes and propagate through the block.

*(figure: damage field on the deformed mesh near peak load; will be added in a follow-up)*


Running and visualizing
-----------------------

.. code-block:: bash

   cd examples/solid/emmentaler

   # Pick one of:
   python emmentaler_elasticity.py    --h 0.08 --steps 11
   python emmentaler_hyperelastic.py  --h 0.08 --load_steps 10
   python emmentaler_phasefield.py    --h 0.1  --load_steps 20

Outputs are written to subdirectories under
``examples/solid/emmentaler/output/``. To inspect:

1. Open the ``.vtk`` file in ParaView.
2. Color by ``von_mises_stress`` (elasticity / hyperelasticity) or
   ``alpha`` (phase-field).
3. Apply ``Warp By Vector`` on ``displacement`` with a 10–20×
   scale factor.
4. Optionally clip with normal :math:`(0, 1, 0)` to expose the
   internal hole structure.


What's next
-----------

* :doc:`hyperelastic_beam` — a smaller / faster Neo-Hookean
  problem to learn the LBFGS pattern in isolation.
* :doc:`hertzian_contact` — another LBFGS-driven problem; here
  the energy comes from contact penalty rather than from
  elasticity.
* :doc:`mousavi2026` — the dataset suite uses the same
  ``NeoHookeanModel`` over many configurations.

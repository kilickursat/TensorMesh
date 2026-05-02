3D Lid-Driven Cavity
====================

The 3D extension of the 2D cavity benchmark. The geometry is a
unit cube :math:`(0, 1)^3` filled with fluid; the top face slides
at unit speed in the :math:`x` direction; all other faces enforce
no-slip. The script ``examples/fluid/cavity_3d/cavity_3d.py``
solves the steady incompressible Navier-Stokes system at
:math:`\mathrm{Re} = 100` on a tetrahedral mesh.

Because the underlying ``NavierStokesAssembler`` (see
:doc:`cavity`) is **dimension-generic** — it takes the spatial
dimension from ``gradu.shape[0]`` and stamps a
``(dim+1) × (dim+1)`` block at each quadrature point — switching
from 2D to 3D is essentially a mesh swap and a slightly larger
DOF mask. There is no separate "3D solver".


Problem
-------

.. math::

   \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u}
   \quad \text{in } \Omega = (0, 1)^3,
   \qquad
   \nabla \cdot \mathbf{u} \;=\; 0,

with :math:`\rho = 1`, :math:`\mu = 1/\mathrm{Re}`, and boundary
conditions

* top face (:math:`y = 1`): :math:`\mathbf{u} = (1, 0, 0)`
* other faces: :math:`\mathbf{u} = \mathbf{0}` (no-slip)
* one node: :math:`p = 0` (pressure pin).


What changes from 2D
--------------------

* ``Mesh.gen_cube(chara_length=…, element_type="tet")`` replaces
  ``Mesh.gen_rectangle``.
* The DOF layout is per-node ``[u, v, w, p]`` instead of
  ``[u, v, p]``: 4 DOFs per node × ``n_points``.
* The lid mask now picks faces of the cube
  (``mesh.points[:, 1] > 1 - eps``) rather than edges of the
  square.
* The output is volumetric: ``cavity_3d.vtu`` for ParaView, plus
  cross-section slices at :math:`z = 0.5` rendered via PyVista as
  ``cavity_3d.png``.

Everything else — the Picard loop, the SUPG/PSPG stamp, the
pressure pin, the use of :class:`~tensormesh.Condenser` — is
identical to :doc:`cavity`. The same ``NavierStokesAssembler``
class works in both cases:

.. code-block:: python
   :caption: examples/fluid/cavity_3d/cavity_3d.py (essence)

   mesh = Mesh.gen_cube(chara_length=1.0/n_grid,
                        element_type="tet").double()
   assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
   condenser = Condenser(u_mask, u_val)

   for i in range(max_iter):
       w_prev = u_full.reshape(-1, 4)[:, :3]              # 3D velocity
       K = assembler(mesh.points, point_data={"w_prev": w_prev})
       f = torch.zeros(n_points * 4, dtype=torch.float64)
       K_, f_ = condenser(K, f)
       u_full = condenser.recover(K_.solve(f_))

The 4-DOFs-per-node layout means the resulting linear system is
roughly :math:`(4 n_\text{nodes})^2` in nonzero count — at
``chara_length=0.05`` that is on the order of a million unknowns,
so a GPU backend (``backend="cudss"`` or ``"pytorch"``, see
:doc:`../../user_guide/linear_solvers`) is recommended once you
go beyond a few thousand nodes.

*(figure: cavity_3d.png — speed magnitude + pressure on the z=0.5 slice; will be added in a follow-up)*


Running it
----------

.. code-block:: bash

   cd examples/fluid/cavity_3d
   python cavity_3d.py     # writes cavity_3d.vtu and cavity_3d.png

Open ``cavity_3d.vtu`` in ParaView for full volumetric inspection
(streamlines, iso-surfaces). The PNG is a quick sanity check.


What's next
-----------

* :doc:`cavity` — the 2D version, more pedagogical detail on the
  ``NavierStokesAssembler`` weak form.
* :doc:`cylinder_flow` — transient flow with implicit Euler.
* :doc:`../../user_guide/linear_solvers` — backend choice for the
  larger 3D systems.

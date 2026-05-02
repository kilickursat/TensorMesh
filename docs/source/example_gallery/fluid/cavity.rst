2D Lid-Driven Cavity
====================

The lid-driven cavity is the textbook benchmark for incompressible
Navier-Stokes solvers. The physics is simple — a square box of
fluid, the top wall slides at unit speed, no-slip on the other
three walls, no body forces — but at moderate Reynolds numbers it
already exhibits a primary vortex plus secondary corner eddies
that any reasonable solver must reproduce. The script
``examples/fluid/cavity/cavity.py`` runs the steady-state problem
at :math:`\mathrm{Re} = 100` on a triangulated unit square.


Problem
-------

The strong form of the steady incompressible Navier-Stokes
equations is

.. math::

   \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u}
   \quad \text{in } \Omega,
   \qquad
   \nabla \cdot \mathbf{u} \;=\; 0,

with :math:`\Omega = (0, 1)^2`, :math:`\mu = 1/\mathrm{Re}`,
:math:`\rho = 1`, and boundary conditions

* top lid (:math:`y = 1`): :math:`\mathbf{u} = (1, 0)`
* other walls: :math:`\mathbf{u} = \mathbf{0}` (no-slip)
* one node: :math:`p = 0` (pressure pin to fix the null space).


Weak form and stabilization
---------------------------

The standard mixed weak form treats velocity and pressure
together. Because the script uses **equal-order** P1-P1 elements
for both, the discrete LBB condition is violated and the naive
Galerkin formulation has spurious pressure modes. The fix is
**SUPG/PSPG stabilization**: add a residual-based perturbation to
both the momentum and the continuity equations, scaled by a
mesh-size-dependent parameter :math:`\tau`.

The implementation is a custom
:class:`~tensormesh.ElementAssembler` defined locally in the
example. Its ``forward`` returns a 3×3 block stamp per quadrature
point — two velocity components plus pressure — that includes the
Galerkin convection / diffusion / pressure-gradient terms and the
SUPG/PSPG additions:

.. code-block:: python
   :caption: examples/fluid/cavity/cavity.py (essence)

   class NavierStokesAssembler(ElementAssembler):
       def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
           self.rho, self.mu, self.tau = rho, mu, tau

       def forward(self, u, v, gradu, gradv, w_prev):
           dim = gradu.shape[0]

           convection = self.rho * torch.dot(w_prev, gradv) * u
           diffusion  = self.mu * torch.dot(gradu, gradv)
           supg_test  = self.tau * torch.dot(w_prev, gradu)
           supg_conv  = self.rho * torch.dot(w_prev, gradv) * supg_test
           k_diag     = convection + diffusion + supg_conv

           rows = []
           for d_test in range(dim):
               row = []
               for d_trial in range(dim):
                   row.append(k_diag if d_test == d_trial
                              else torch.tensor(0.0))
               row.append(-v * gradu[d_test]
                          + self.tau * gradv[d_test] * torch.dot(w_prev, gradu))
               rows.append(torch.stack(row))

           cont_row = []
           for d_trial in range(dim):
               cont_row.append(gradv[d_trial] * u
                               + self.tau * self.rho
                                 * torch.dot(w_prev, gradv) * gradu[d_trial])
           cont_row.append(self.tau * torch.dot(gradv, gradu))   # PSPG
           rows.append(torch.stack(cont_row))
           return torch.stack(rows)


Picard iteration
----------------

The convection term :math:`(\mathbf{u}\cdot\nabla)\mathbf{u}` is
linearized by passing the previous-iterate velocity
:math:`\mathbf{w}^{n}` as ``w_prev`` to the assembler:

.. math::

   \rho\, (\mathbf{w}^{n} \cdot \nabla)\mathbf{u}^{n+1}
   \;=\; -\nabla p^{n+1} + \mu\, \Delta \mathbf{u}^{n+1}.

This converges geometrically for moderate Reynolds numbers; the
script iterates until ``||u_new - u_full|| / ||u_new|| < 1e-4``,
typically ~10 iterations at Re=100.

.. code-block:: python

   assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
   condenser = Condenser(u_mask, u_val)

   for i in range(max_iter):
       w_prev = u_full.reshape(-1, 3)[:, :2]
       K = assembler(points, point_data={"w_prev": w_prev})
       f = torch.zeros(n_points * 3, dtype=torch.float64)
       K_, f_ = condenser(K, f)
       u_new  = condenser.recover(K_.solve(f_))

       diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
       u_full = u_new
       if diff < tol:
           break

A few details that matter:

* **DOF layout.** The unknowns are interleaved per node:
  ``[u_0, v_0, p_0, u_1, v_1, p_1, …]``. The
  :class:`~tensormesh.Condenser` ``u_mask`` flattens the
  per-node-per-component Dirichlet pattern into a single 1D
  boolean.
* **Pressure pin.** ``u_mask[2] = True; u_val[2] = 0.0`` clamps
  pressure to zero at node 0 — required because pressure is only
  determined up to an additive constant in incompressible flow.
* ``w_prev`` is passed by name in ``point_data`` and arrives in
  ``forward`` as the matching keyword argument; see
  :doc:`../../user_guide/forms` for the dispatch contract.

.. figure:: /_static/fluid/cavity_results.png
   :alt: Lid-driven cavity speed magnitude and pressure at Re=100
   :width: 100%

   Output of ``cavity.py`` at Re = 100. Left: speed magnitude
   :math:`\|u\|` — the moving lid drags fluid into the upper
   right, sweeping it down the right wall and forming the
   primary vortex. Right: pressure field, with the
   characteristic high-pressure spot in the upper-right corner
   where the lid stagnates against the wall.


Running it
----------

.. code-block:: bash

   cd examples/fluid/cavity
   python cavity.py            # writes cavity_results.png

Console output reports relative residual at each Picard step plus
the final convergence message. Compare the stream-function
contours qualitatively against the Ghia / Ghia / Shin reference
(1982) for Re=100 — the location of the primary vortex center
should match to within a few percent.


What's next
-----------

* :doc:`cavity_3d` — same physics on a tetrahedral cube.
* :doc:`cylinder_flow` — adds transient time stepping.
* :doc:`taylor_green` — the same family of assemblers, now used
  for a quantitative convergence study against an exact solution.
* :doc:`../../user_guide/forms` — argument-dispatch contract for
  vector-valued ``forward`` returns.

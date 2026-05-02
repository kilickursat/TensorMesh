Cylinder Flow (Vortex Shedding)
================================

The transient counterpart to the steady cavity examples. The
script ``examples/fluid/cylinder_flow/cylinder_flow.py`` runs the
classic *flow past a circular cylinder* benchmark — a long
rectangular channel with a small cylinder near the inlet. At
:math:`\mathrm{Re} = 100` the wake becomes unstable, vortices
shed alternately from the top and bottom of the cylinder, and a
**von Kármán vortex street** propagates downstream. The geometry
and parameters follow the DFG 2D-1 benchmark (Schäfer & Turek,
1996), which gives reference values for drag/lift coefficients
and the Strouhal number.


Problem
-------

The transient incompressible Navier-Stokes system,

.. math::

   \rho\, \frac{\partial \mathbf{u}}{\partial t}
   + \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u},
   \qquad \nabla \cdot \mathbf{u} = 0,

on the channel :math:`\Omega = [0, 2.2] \times [0, 0.41]` with a
circular cylinder of radius :math:`r = 0.05` centered at
:math:`(0.2, 0.2)`. Boundary conditions:

* inlet (:math:`x = 0`): parabolic profile
  :math:`u_x(y) = 4\,U_\text{max}\, y\, (H - y) / H^2`,
  :math:`u_y = 0`,
* walls and cylinder surface: no-slip,
* outlet (:math:`x = 2.2`): "do-nothing" + pressure pin,

with :math:`U_\text{max} = 1.5` giving :math:`\bar{U} = 1`,
:math:`D = 0.1`, :math:`\rho = 1`, :math:`\mu = 10^{-3}`, and
hence :math:`\mathrm{Re} = \rho \bar{U} D / \mu = 100`.


Time integration: implicit Euler
--------------------------------

The transient term :math:`\rho\, \partial_t \mathbf{u}` is
discretized with backward Euler — implicit, unconditionally
stable. Each timestep solves

.. math::

   \rho\, \frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t}
   + \rho\, (\mathbf{u}^{n} \cdot \nabla)\mathbf{u}^{n+1}
   \;=\; -\nabla p^{n+1} + \mu\, \Delta \mathbf{u}^{n+1},

(convection linearized by lagging the advecting velocity to
:math:`\mathbf{u}^n`). The script implements this as a transient
variant of the cavity assembler, with the mass term
:math:`\rho/\Delta t \cdot u\, v` added to the diagonal block
along with a corresponding SUPG residual:

.. code-block:: python
   :caption: examples/fluid/cylinder_flow/cylinder_flow.py (essence)

   class NavierStokesTransientAssembler(ElementAssembler):
       def __post_init__(self, rho=1.0, mu=0.01, dt=1e-3, tau=1e-3):
           self.rho, self.mu, self.dt, self.tau = rho, mu, dt, tau

       def forward(self, u, v, gradu, gradv, w_prev):
           dim = gradu.shape[0]
           mass        = self.rho * (u * v) / self.dt
           convection  = self.rho * torch.dot(w_prev, gradv) * u
           diffusion   = self.mu * torch.dot(gradu, gradv)
           supg_test   = self.tau * torch.dot(w_prev, gradu)
           supg_res    = (self.rho/self.dt * v
                          + self.rho * torch.dot(w_prev, gradv)) * supg_test
           momentum_diag = mass + convection + diffusion + supg_res
           # …assemble (dim+1)×(dim+1) block as in cavity.py…

The previous-timestep solution :math:`\mathbf{u}^n` enters the
RHS through a separate ``NodeAssembler`` (``MomentumRHSAssembler``)
that contributes :math:`\rho/\Delta t\, \mathbf{u}^n \cdot v`.
Both assemblers share the same ``tau`` field, updated each step
to reflect the local cell Reynolds number.


Stabilization
-------------

The cylinder problem is more demanding than the cavity:
upstream cells see nearly-uniform flow, the wake region needs
fine-grained convection-dominated stabilization. The script uses
an **adaptive** :math:`\tau`:

.. math::

   \tau \;=\;
   \left[
     \left(\frac{2}{\Delta t}\right)^2
     + \left(\frac{2|\mathbf{u}|}{h}\right)^2
     + \left(\frac{4\nu}{h^2}\right)^2
   \right]^{-1/2},

evaluated per element from the local mesh size :math:`h` and
velocity magnitude. This makes the diffusive part of SUPG dominate
where the flow is slow and the convective part dominate in the
wake.


Post-processing
---------------

Two derived fields are computed at every saved frame:

* **Vorticity** :math:`\omega = \partial_x v - \partial_y u` —
  computed by L² projection: assemble a mass matrix
  :class:`~tensormesh.MassElementAssembler`, project
  :math:`\partial_x v - \partial_y u` evaluated on quadrature
  points back onto the nodal basis, solve
  :math:`M \omega = \tilde\omega`. Standard postprocessing recipe
  for piecewise-linear FEM.
* **Drag and lift coefficients** on the cylinder surface — the
  script integrates the surface traction
  :math:`\mathbf{t} = (\sigma \cdot \mathbf{n})` over the cylinder
  facets via a :class:`~tensormesh.FacetAssembler` instance.

For a Strouhal number :math:`\mathrm{St} = f D / \bar{U}` of about
0.30, the script's plotted lift coefficient should oscillate at
roughly :math:`f \approx 3` Hz once the wake stabilizes
(typically after :math:`t \gtrsim 5`).


Output and rendering
--------------------

* **Frame sequence.** Every ``N`` steps the script writes a VTU
  with the velocity, pressure, and vorticity fields. ParaView
  opens the whole sequence as a time series.
* **MP4 rendering.** The companion script
  ``examples/fluid/cylinder_flow/render_video.py`` rasterizes the
  VTU sequence into ``cylinder_flow.mp4`` via PyVista + ffmpeg.
* **Final snapshot.** ``cylinder_flow_final.png`` is a three-panel
  static figure (vorticity, speed, pressure) of the last step,
  for inclusion in talks and reports.

*(figure: vorticity field at t = 8 s showing the von Kármán street; will be added in a follow-up)*


Running it
----------

.. code-block:: bash

   cd examples/fluid/cylinder_flow
   python cylinder_flow.py            # writes vtu sequence + final.png
   python render_video.py             # writes cylinder_flow.mp4

The transient run is the longest in the gallery — the default
configuration is several thousand timesteps. Reduce ``n_steps`` or
coarsen the mesh for a quick smoke test.


What's next
-----------

* :doc:`cavity` — the steady cousin with the same SUPG/PSPG
  recipe.
* :doc:`taylor_green` — a transient problem with an exact
  solution, for verifying the time integrator's order.
* :doc:`flow_obstacles` — steady flow through a more complex
  channel geometry.
* :doc:`../../user_guide/time_integration` — implicit-linear
  integrators that subsume the manual backward-Euler loop.

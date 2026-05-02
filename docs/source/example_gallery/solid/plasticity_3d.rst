Plasticity 3D
=============

The 3D extension of :doc:`plasticity_strip`: J2 flow theory with
linear isotropic hardening on a :math:`0.5 \times 0.5 \times 0.5` m
steel cube, stretched 40 % in the :math:`x` direction over 50
loading steps and unloaded over 50 more. The script is
``examples/solid/plasticity_3d/plasticity_3d.py``.

The constitutive model, the variational-update pattern, and the
history-variable bookkeeping are **identical** to the 2D version —
:class:`~tensormesh.assemble.J2Plasticity` is dimension-generic.
What changes here is scale, so the page focuses on what's actually
different rather than re-stating the J2 model.


Differences from the 2D strip
-----------------------------

* **3D mesh.** ``gen_cube(chara_length=0.04, ...)`` produces a
  tetrahedral cube with on the order of 5–10 k DOFs (vs. ~3 k in
  the 2D strip). Each L-BFGS iteration assembles and integrates
  over significantly more quadrature points.
* **More load steps.** 50 loading + 50 unloading = 100 steps,
  reaching 40 % strain — far enough into plasticity that the
  hardening branch dominates.
* **Larger plastic-strain magnitude.** At 40 % nominal strain the
  small-strain assumption is being stretched (literally); the
  example is still useful as a regression test of the J2 model
  but you would not trust the absolute stress numbers far from
  the elastic regime.
* **Different output.** ``plasticity_3d.mp4`` shows the deformed
  cube colored by **von Mises stress** plus a force-displacement
  curve at the loaded face.


Code skeleton
-------------

.. code-block:: python
   :caption: examples/solid/plasticity_3d/plasticity_3d.py (essence)

   from tensormesh.assemble import J2Plasticity
   from tensormesh.material import IsotropicMaterial

   mesh = gen_cube(chara_length=0.04,
                   left=0, right=0.5, bottom=0, top=0.5,
                   front=0, back=0.5)
   steel = IsotropicMaterial("Steel_Hardening",
                             E=200e9, nu=0.3,
                             sigma_y=250e6, H=1e9)
   model = J2Plasticity.from_mesh(mesh, material=steel)

   u = torch.zeros_like(mesh.points, requires_grad=True)
   optimizer = optim.LBFGS([u], lr=1.0, max_iter=50,
                           history_size=50,
                           line_search_fn="strong_wolfe")

   for step in range(1, n_total_steps + 1):
       target_disp = piecewise_load_schedule(step)
       def closure():
           # …same as plasticity_strip, with 3D BCs…
           pass
       optimizer.step(closure)
       model.advance_history(...)

The boundary conditions follow the same pattern as the 2D strip:
roller on the :math:`x = 0` face, corner pin at the origin,
prescribed :math:`u_x` on the :math:`x = 0.5` face.


Performance notes
-----------------

* The 3D problem is L-BFGS-bound — most of the wall-clock time
  is in the per-iteration energy evaluation + autograd backward.
  GPU acceleration helps significantly: ``mesh.to("cuda")`` and
  ``u.to("cuda")`` move the entire optimization to the device.
* History-variable updates (``model.advance_history``) are cheap
  compared to the per-step L-BFGS sweep; do not worry about
  optimizing them.
* If memory becomes an issue at finer mesh resolutions, set the
  assembler's ``batch_size`` argument to chunk the
  per-quadrature-point integrand tensor — see
  :doc:`../../user_guide/batched_workflows` for the memory
  story.

*(figure: deformed cube colored by von Mises stress at peak strain; will be added in a follow-up)*


Running it
----------

.. code-block:: bash

   cd examples/solid/plasticity_3d
   python plasticity_3d.py     # writes plasticity_3d.mp4


What's next
-----------

* :doc:`plasticity_strip` — the 2D version with the full
  exposition of the variational update.
* :doc:`emmentaler` — a 3D problem with hyperelasticity (no
  plastic history) plus phase-field damage.
* :doc:`mousavi2026` — Neo-Hookean hyperelasticity at industrial
  dataset scales.

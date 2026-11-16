# Prompt: draft the ME-474 report with Claude

Start Claude Code in a clone of the repository (branch `me474-2d-study`) and paste everything below the line.
In a Claude chat instead, attach `ME474/PLAN.md`, `ME474/RESULTS_OUTLINE.md`, `ME474/README.md`,
`ME474/setup_fluent.py`, `ME474/fluent_points.csv`, `ME474/runs/summary.csv`, `ME474/runs/drag_curve.csv`,
`ME474/runs/drag_curve_cd_nose.csv`, `ME474/runs/residuals.csv`, `ME474/figures/fig_cd_mach.pdf`, the Fluent
figures you want discussed, the course checklist / report template (`Checklist-Report_template.pdf`, E. Boujo) and
last year's example report (Assessment 2, December 2025, NACA 4412 in ground effect), both on the shared Drive.

---

You are helping group 13 write their report for ME-474 Numerical Flow Simulation (EPFL), Assessment 2: a 2D
axisymmetric RANS study in Ansys Fluent 2024 R2 of the drag of two 720 mm rocket nose cones. One PDF per group, due
1 January 2027; our internal deadline is 28 December 2026. All simulations are finished. This task is writing only:
do not start Fluent runs or change any script, mesh or setting.

## The two questions the report answers

1. At which Mach number does the drag rise begin for each nose (defined as dC_D/dM = 0.1)?
2. Which nose has the lower drag, and over which Mach range?

## Sources, in this order. Every number in the report must come from them

1. `ME474/RESULTS_OUTLINE.md`: the Results section, subsection by subsection, with every number, its uncertainty
   and the figure to show. Section "Results" follows it.
2. `ME474/PLAN.md`: the whole study. Section 2c-2f mesh study (GCI), 2g domain study, 2h validation (sharp cone
   against Taylor-Maccoll), 2i solver cross-check and production, 5 mesh settings, 6 domains, 7 decisions,
   9 points to mention in the report, 10 the figures.
3. `ME474/README.md`, `ME474/setup_fluent.py`: models, boundary conditions, solver schedules, convergence criterion.
4. `ME474/fluent_points.csv`: free stream at each Mach number (from design A's simulated ascent, used for both).
5. `ME474/runs/summary.csv` (one row per run), `runs/drag_curve.csv` (headline C_D, nose + coupler),
   `runs/drag_curve_cd_nose.csv` (nose alone), `runs/residuals.csv` (orders of magnitude per equation).
6. `ME474/figures/`: `fig_cd_mach.pdf` and the native Fluent PNGs (names in RESULTS_OUTLINE.md and PLAN.md section 10).
7. `ME474/gci.py`: the GCI procedure (Celik et al. 2008).
8. `ME474/runs/mesh_sizes.csv`: cells, nodes and faces of every mesh (read from the case files by `mesh_sizes.py`).

If a number you need is not in these files, write `[TODO: ...]` in the text and list it for us. Never estimate,
round differently or recompute a value in another way than the files do.

## Fixed facts and decisions (state them, do not reopen them)

* Designs: A = LV-Haack nose 679 mm + 41 mm cylindrical coupler; B = Von Karman nose 700 mm + 20 mm coupler.
  Base diameter D = 180 mm, tip radius 4.5 mm, both 720 mm long to the body.
* Physics: steady, 2D axisymmetric, RANS k-omega SST; air as ideal gas with Sutherland viscosity; energy on;
  adiabatic walls.
* Boundary conditions: inlet and far field are pressure far-field at the free stream of `fluent_points.csv`,
  turbulence intensity 1 % and viscosity ratio 10; pressure outlet at p_inf with backflow total temperature = free-stream
  T0. Reference values from the inlet; A_ref = pi (0.09 m)^2 = 0.025447 m^2, reference length 0.18 m.
* Numerics: hybrid initialisation. Pressure-based Coupled (pseudo-time) up to M 1.10: 200 iterations with first-order
  upwind (pressure second order), then 1000 second order. Density-based Roe-FDS with least-squares gradients from
  M 1.20: 500 iterations first order at CFL 1, 500 at CFL 5, then 3000 second order at CFL 5. After the schedule,
  500-iteration blocks until the nose C_D changes by less than 0.1 % over the last 500 iterations. Residuals are
  monitored, not a stop criterion.
* Production set: medium mesh (decision 1); 40 D domain up to M 1.10, 20 D at M 1.20 and burnout (decision 2);
  9 Mach points 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20, 1.465 (burnout).
* Drag-rise definition (decision 3): dC_D/dM from finite differences, the slope of each interval at its midpoint,
  the crossing of 0.1 interpolated linearly.
* Solver at M 1.20 (decision 6): density-based. The pressure-based runs at M 1.20 (A and B) are a cross-check and
  give the wall Cp(x) figures, because the density-based wall pressure zigzags there.
* Headline C_D (decision 7): nose + coupler, x = 0 to 720 mm for both designs. The nose alone is given in the table.

## Explain where the inputs come from

The reader must be able to see where every boundary condition and reference value comes from. Explain each of these
in the report (in the section the checklist puts it in), from the files named; if a file does not say why, write
`[TODO: why ...]` rather than inventing a reason.

* **Free stream** (section 4): the 9 points are design A's simulated ascent, from the Rocket Team's trajectory model
  (`ME474/make_points.py` calls `studies/total_length.py` with `configs/weisshorn_650.yaml`; flight model in
  `nosecone/flight.py`), with the ISA troposphere (`nosecone/config.py`, `AtmosphereConfig`). For each Mach number:
  altitude, p, T, rho, V, q, Re_L, T0, p0 (`fluent_points.csv`). Design A's points are used for both shapes so that
  A against B compares shapes only (`make_points.py`). The values use Fluent's default air (MW 28.966, cp 1006.43
  J/(kg K), three-coefficient Sutherland: `CFD/studyA/freestream_bc.py`), so they match what Fluent computes.
* **Turbulence at the boundaries**: intensity 1 % and viscosity ratio 10 at inlet, far field and outlet backflow,
  i.e. k = 1.5 (I V)^2 and omega = rho k / (mu * 10) (`CFD/studyA/freestream_bc.py`; k and omega per point in
  `fluent_points.csv`).
* **Boundary types** (`ME474/setup_fluent.py`): pressure far-field at inlet and far field, pressure outlet at p_inf
  with backflow total temperature = free-stream T0 (not T_inf), axis, adiabatic no-slip walls. Reference values from
  the inlet zone, A_ref = pi (0.09 m)^2, reference length D = 0.18 m.
* **First-cell height and y+**: the flat-plate estimate in `freestream_bc.first_cell_height` and the resulting y+
  (PLAN.md section 5, `summary.csv` column `yplus_max`).
* **Taylor-Maccoll validation** (section 7.8, summed up in 9.2): what the Taylor-Maccoll equation is (inviscid,
  axisymmetric conical flow behind an attached conical shock), how `nosecone/taylor_maccoll.py` solves it (the
  conical-flow ODE integrated inwards from a trial shock angle, starting from the oblique-shock relations, the shock
  angle found by root finding so that the flow meets the cone surface; surface Cp and Mach from the solution), the
  7.1° sharp cone, the free stream used (burnout and M 1.20), the targets (Cp 0.0723 and 0.0859, shock angles 43.3°
  and 56.7°) and how Fluent was compared: wall Cp over x = 50-650 mm, shock angle with `ME474/shock_angle.py`
  (PLAN.md 2h). The targets were computed on our PC with scipy (PLAN.md section 1).
* **GCI**: the Celik et al. (2008) procedure as implemented in `ME474/gci.py` (h from the cell count, apparent order,
  Richardson extrapolation, safety factor), and why B at M 0.95 has no GCI (oscillatory convergence).
* **Drag-rise criterion** dC_D/dM = 0.1 and how it is evaluated (decision 3, `drag_curve.py`).
* **Checks of each run**: the columns of `check_run.py` (README.md, "Checks per run"), e.g. the pitot pressure p0
  behind a normal shock as the upper bound of the wall pressure.

## Key results (already in RESULTS_OUTLINE.md; check against it)

* Drag rise: A at M 0.950, B at M 0.937 (about +-0.01 from the point spacing). B rises earlier and more steeply.
* B has the lower drag except between M 0.958 and 1.019, where it is up to 6.3 % higher (M 0.98). B is 2.7 % lower
  at M 0.95, 13.6 % at M 1.20 and 15.9 % at burnout.
* B's friction is 6-7.5 % lower everywhere (7.6 % less wetted area); the whole swing is pressure drag.
* The A-B differences are well above the numerical uncertainty except at M 1.02 (equal); see R.3 and R.6 for the
  bands and the limits (M 1.00 is not domain-converged, B's domain sensitivity at M 1.00 was not tested).

## Structure: the course checklist / report template, section by section

Last year's example report follows the checklist one to one (title page, abstract, contents, acronyms and symbols,
chapters 1-9, references, one appendix; about 33 pages). Use the same numbering; a sub-point with nothing to say gets
one sentence, not a gap.

* Front matter: title page (ME-474 Numerical Flow Simulation, Assessment 2, title, date, group 13 with names and
  Sciper numbers: `[TODO]`), abstract (one paragraph: the questions, the method, the answers with numbers), table of
  contents, list of acronyms and of symbols with units.
1. Introduction. 1.1 Context and goals: the vehicle in one paragraph, the two designs, the two questions; literature
   only from references we give you. 1.2 Type of analysis, methodology: 2D axisymmetric steady RANS at 9 Mach points,
   mesh and domain studies, validation, the data available (`fluent_points.csv`).
2. Geometric modelling and hypotheses. 2.1 Geometry: both profiles, dimensions, couplers, tip radius, and the
   simplifications (body continued to the outlet, no fins, zero angle of attack). 2.2 Unit system: SI. 2.3 Characteristic
   dimensions and dimensionless numbers: D, L = 0.72 m, M_inf, Re_L (`fluent_points.csv`). 2.4 Symmetry: axisymmetric
   at zero angle of attack (geometry and boundary conditions). 2.5 Geometrical space: 2D axisymmetric; the 10 / 20 / 40 D
   domains.
3. Physical modelling and hypotheses. 3.1 Physical behaviour: compressible ideal gas, Newtonian, Sutherland viscosity,
   energy equation, steady. 3.2 Fluid properties with values and units. 3.3 Turbulence: k-omega SST, wall-resolved
   (y+ < 1).
4. Boundary conditions, external forces, initial conditions. 4.1 Boundary conditions: type, location, values, units
   (table), and the free stream of the 9 points (table). 4.2 External forces: gravity neglected (say why in one
   sentence). 4.3 Initial conditions: steady problem, hybrid initialisation.
5. Computational mesh. 5.1 Mesh type: hybrid. 5.2 Cell types: quadrilateral inflation layers, triangles elsewhere.
   5.3 Sizes and local refinements, boundary-layer mesh, total number of cells and nodes (PLAN.md section 5,
   `runs/mesh_sizes.csv`). 5.4 Final (medium) mesh: pictures and quality metrics.
6. Numerical methods. 6.1 Finite volumes; spatial orders. 6.2 Solvers: pressure-based coupled and density-based Roe;
   variables solved for. 6.3 Solution options: initialisation, schedules, convergence criterion. 6.4 Computed
   quantities: C_D on wall_nose + wall_coupler, reference values, C_D,p and C_D,v, wall Cp and Cf, dC_D/dM.
7. Mesh size / domain size convergence study. 7.1 Criterion: C_D of the nose. 7.2 The meshes and domains: pictures,
   table of cells. 7.3 Results on the coarsest mesh / smallest domain. 7.4 Results on the finest mesh / largest domain.
   7.5 Relative error: GCI (Celik et al. 2008), domain change. 7.6 Choice: medium mesh, 40 D / 20 D. Then 7.7 iterative
   convergence and 7.8 validation against Taylor-Maccoll (summed up again in 9.2).
8. Results: follow RESULTS_OUTLINE.md, R.0 to R.7.
9. Analysis and conclusions. 9.1 Summary of the results. 9.2 Accuracy and uncertainty: the limitations with their size,
   verification and validation. 9.3 Criteria and the comparison of A and B (drag-rise criterion dC_D/dM = 0.1).
   9.4 Conclusion: answer the two questions. 9.5 Recommendations, for the simulation and for the vehicle.
* References: only sources we give you or that the files name (e.g. Celik et al. 2008); never invent one, write
  `[REF TODO: what it must support]`.
* Appendix: only study material (e.g. the residual plots of all runs). No appendix about the Rocket Team project and
  none about the use of Claude or other AI tools.

## Figures

* The assignment asks for readable figures: axis labels, suitable ranges, scales and fonts.
* Use the figures in `ME474/figures/` as they are. They stay as Fluent draws them (2400 x 1500, Fluent's fonts, which
  are small): place the Fluent plots at full text width, not side by side, so their text stays readable. The Fluent figures are native Fluent plots on purpose: do not
  redraw them in matplotlib. Captions: which run (shape, Mach number, solver), what the reader should see.
* Figures for the sections before Results (PLAN.md section 10, last item; all native Fluent):
  * Geometry: `fluent_geometry_profiles`, `fluent_geometry_tip` (4.5 mm tip sphere), `fluent_geometry_domains`
    (10 / 20 / 40 D). The dimensions go in the caption or the text: the plots have no dimension lines.
  * Mesh: `fluent_mesh_domain_A_D40`, `fluent_mesh_nose_A_medium`, `fluent_mesh_tip_A_medium`,
    `fluent_mesh_tipzoom_A_medium`, `fluent_mesh_wall_A_medium`, `fluent_mesh_shoulder_A_medium` (and the same for
    B_medium; A coarse and fine for the mesh study). Each has Fluent's scale bar at the bottom centre; PLAN.md
    section 10 also lists the width of each view.
  * Convergence: `fluent_meshstudy_M0.95`, `fluent_meshstudy_change`, `fluent_domainstudy_M0.95`,
    `fluent_domainstudy_M1.00`, `fluent_domainstudy_change`; validation `fluent_cp_cone_burnout`, `fluent_cp_cone_M1.20`,
    `fluent_mach_wide_cone_burnout`; the sliver fan at the cone's sharp tip: `fluent_mesh_tipzoom_cone`.
  * Every Fluent XY plot carries Fluent's own title, the plotted functions ("pressure-coefficient" on the Results
    plots, "nose-drag-coefficient vs. relative-cell-size" on the study plots). Leave it; the caption says what the
    figure shows, as for any figure.
* If a figure you need is not there, put a `[FIGURE TODO: ...]` placeholder with what it must show, and list it for us.

## Writing rules

* Concise technical English; past tense for what was done, present tense for what the results show.
* Every comparison of A and B with its uncertainty and the reason the difference is (or is not) significant.
* Report the limitations honestly, with their size (PLAN.md section 9 and R.6): stagnation pressure 0.09-0.90 % above
  theory in the pressure-based runs at M 0.90-1.10 (about 1e-4 in C_D); M 1.00 not domain-converged; the Roe wall-pressure zigzag and the residual stall
  at 2.2-2.4 orders at M 1.20; the k residual creeping back up from M 1.05 while C_D stays flat; coarse-mesh
  oscillation at burnout; tip slivers in the cone mesh; GCI refinement not in the wall-normal direction.
* Notation: C_D, C_D,p, C_D,v, Cp, Cf, M_inf, y+; SI units; A for LV-Haack, B for Von Karman throughout.
* No page limit is set: complete but concise.
* Use the ERT LaTeX template: `CFD/source/main.tex` with `CFD/source/pictures/template/` (title page background,
  ERT logo and patch, red header and footer rules, Helvetica). Make `ME474/report/main.tex` from it with one file per
  chapter in `ME474/report/sections/`, and point `\graphicspath` at `../figures/`. Replace the title-page text (title,
  names with Sciper numbers, title line, doc ID, date) and the header line for this report; ask us for the doc ID
  (`[TODO]`). Keep the template's layout as it is.

## Start

First give us the section plan with the figure placement and a page budget per section, and the list of `[TODO]`
items and missing figures you already see. Wait for our answer, then write the sections one by one, starting with
Results.

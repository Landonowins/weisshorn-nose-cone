# ME-474 group 13: outline of the Results section

Draft of 16 Nov 2026. Every number comes from `runs/summary.csv` (`python check_run.py`) and `runs/drag_curve.csv`
(`python drag_curve.py`); every figure is in `figures/` (PLAN.md section 10 says how to remake them). The mesh study,
the domain study and the cone validation belong in the convergence section before this one; Results only cites them.

Decisions behind the numbers (PLAN.md section 7):
* **Headline C_D = nose + coupler** (decided 16 Nov): x = 0 to 720 mm for both designs, A = LV-Haack 679 mm nose +
  41 mm coupler, B = Von Karman 700 mm + 20 mm. That is the part that differs between the designs; behind x = 720 mm
  the body is the same. The nose alone (`wall_nose`) is given as well, in the table.
* Medium mesh; 40 D domain up to M 1.10, 20 D at M 1.20 and burnout; pressure-based Coupled up to M 1.10,
  density-based Roe from M 1.20 (decided 15 Nov).
* Drag-rise Mach number: where dC_D/dM = 0.1 (decided 14 Nov), from finite differences: the slope of each interval
  at its midpoint, the crossing of 0.1 interpolated linearly between midpoints.

## R.0 Quantities (short box at the start)

* C_D = F_x / (q_inf A_ref), A_ref = pi (0.09 m)^2 = 0.025447 m^2. Forces from Fluent's wall-forces report,
  x direction, pressure + viscous; `wall_nose` + `wall_coupler` (the coupler is a cylinder: friction only).
* 9 Mach points: 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20 and 1.465 (burnout), free stream from the
  trajectory (`fluent_points.csv`). 18 production runs.

## R.1 Drag curves

Figure: `fig_cd_mach` (a). Table (nose + coupler; pressure part all on the nose):

| M | A | B | B vs A | A pressure / viscous | B pressure / viscous | nose only: B vs A |
|---|---:|---:|---:|---|---|---:|
| 0.90 | 0.03334 | 0.03112 | -6.7 % | 0.00219 / 0.03115 | 0.00205 / 0.02907 | -3.8 % |
| 0.95 | 0.03352 | 0.03262 | -2.7 % | 0.00269 / 0.03082 | 0.00368 / 0.02894 | -0.1 % |
| 0.98 | 0.03832 | 0.04074 | +6.3 % | 0.00737 / 0.03096 | 0.01207 / 0.02867 | +9.6 % |
| 1.00 | 0.05485 | 0.05685 | +3.6 % | 0.02422 / 0.03063 | 0.02853 / 0.02831 | +5.8 % |
| 1.02 | 0.07568 | 0.07555 | -0.2 % | 0.04551 / 0.03018 | 0.04764 / 0.02791 | +1.2 % |
| 1.05 | 0.08656 | 0.08216 | -5.1 % | 0.05681 / 0.02975 | 0.05464 / 0.02752 | -4.0 % |
| 1.10 | 0.09364 | 0.08417 | -10.1 % | 0.06445 / 0.02919 | 0.05712 / 0.02704 | -9.3 % |
| 1.20 | 0.09696 | 0.08379 | -13.6 % | 0.06868 / 0.02828 | 0.05750 / 0.02629 | -12.8 % |
| 1.465 | 0.09526 | 0.08009 | -15.9 % | 0.06904 / 0.02622 | 0.05568 / 0.02441 | -15.3 % |

Message: C_D is flat at about 0.033 up to M 0.95, rises steeply through M 1 to about 0.09 by M 1.05-1.10, then stays
flat or falls slightly to burnout. Up to M 0.95 it is almost all friction (pressure part 7-11 %); from M 1.05 the
pressure part is 65-72 %.

## R.2 Drag rise (key question 1)

Figure: `fig_cd_mach` (b).
* **A at M 0.950, B at M 0.937** (about +-0.01 from the point spacing). Nose alone: 0.951 / 0.937; the couplers
  add an almost constant friction, so the slope hardly changes.
* B's rise starts earlier and is steeper at first: slope 0.27 against 0.16 between M 0.95 and 0.98. Both peak just above
  M 1 (A 1.04, B 0.94 between M 1.00 and 1.02) and fall below 0.1 again by M 1.10 (B) and 1.15 (A).
* Robust to the domain: the crossing lies below M 0.98, where 40 D is within about 0.4 % of an infinite domain
  (domain study at M 0.95).

## R.3 Which shape has lower drag (key question 2)

Figure: `fig_cd_mach` (a), band shaded.
* **B is lower everywhere except M 0.958-1.019**, where its earlier rise puts it up to 6.3 % above A (M 0.98).
  B is 6.7 % lower at M 0.90, 2.7 % at M 0.95, 10.1 % at M 1.10, **13.6 % at M 1.20 and 15.9 % at burnout**.
* Significance, against the mesh uncertainty of the A-B difference at M 0.95 (+-0.00036, from the mesh studies of
  A and B; the coupler friction is mesh-independent):
  * M 0.95: A - B = +0.00090, 2.5 times the band: B is lower. (Nose alone, +0.00002: equal.)
  * M 0.98 and 1.00: A - B = -0.0024 / -0.0020, 5-7 times the mesh band. The domain uncertainty at M 1.00 is several
    per cent of C_D (C_D still rises from 20 to 40 D, domain study), but at M 0.95 it was common to A and B
    (-1.47 / -1.41 % from 20 to 40 D); B's domain sensitivity at M 1.00 was not tested. Say so.
  * M 1.02: -0.2 %, equal within the uncertainty: the second crossover is at about M 1.02.
  * Supersonic: A - B = 0.013-0.015, far above the 0.02 % mesh uncertainty at burnout.
* Consistent with the design intent of the Haack series: Von Karman (C = 0) minimises wave drag for a given length
  and base diameter, LV-Haack (C = 1/3) for a given length and volume. Agrees with the earlier 720 mm study
  (-12.8 / -15 % at M 1.20 / burnout, nose alone).

## R.4 Pressure and viscous parts

Figures: `fig_cd_mach` (c), (d); Cf(x): `fluent_cf_AB_<point>`.
* **Viscous: B is 6.1-7.5 % lower at every Mach number** (nose alone 3.5-4.5 %). The reason is the wetted area:
  0.2723 against 0.2948 m^2 over x = 0-720 mm (-7.6 %; noses alone 0.2610 / 0.2716 m^2, couplers 0.0113 / 0.0232).
  The mean Cf along the nose is nearly the same (0.00274 / 0.00276 at M 0.98).
* **Pressure: the whole swing.** B vs A: -6 % at M 0.90, +37 % at 0.95, +64 % at 0.98, +18 % at 1.00, +5 % at 1.02,
  -4 % at 1.05, -11 % at 1.10, -16 % at 1.20, -19 % at burnout.
* A's coupler friction rises at M 0.98-1.00 (0.0020 -> 0.0022): the flow accelerates past the shoulder. Small bump in
  (d); physical.

## R.5 Why: flow fields and wall pressure

* **M 0.98, where B is higher** (`fluent_mach_A_D40_M0.98`, `fluent_mach_B_D40_M0.98`, `fluent_cp_AB_M0.98`):
  B has the stronger supersonic pocket at the shoulder (about M 1.12 against 1.07) and a sharper terminating shock
  just behind it. Its wall Cp is lower than A's only near the tip (x < 0.1 m) and higher from x = 0.1 to 0.69 m.
* **M 1.02-1.10** (`fluent_mach_wide_*`, `fluent_mach_*`): the bow shock is detached and moves in, 146 / 50 / 18 mm
  ahead of A's tip at M 1.02 / 1.05 / 1.10 (7 mm at M 1.20, 3 mm at burnout; from `<run>_axis.csv`).
* **Burnout, where B is lower** (`fluent_cp_AB_burnout`, `fluent_cpfield_*_burnout`): B's Cp is lower over the front
  0.2 m, where the wall is steepest, and slightly higher behind: the opposite balance to M 0.98.
* **M 1.20** (`fluent_cp_AB_M1.20_pressure`, `fluent_cp_solver_A_M1.20`, `fluent_cp_solver_B_M1.20`): the
  density-based wall Cp zigzags from face to face (A at x = 0.03-0.06, 0.38-0.41, 0.57-0.61 m; B from 0.27 to
  0.54 m), around the right mean. Show Cp(x) from the pressure-based runs, which are smooth, and keep the
  density-based C_D (solver decision of 15 Nov). [B pressure-based run of 16 Nov: result to fill in.]
* Tip close-ups (`fluent_mach_tip_*`): the subsonic region behind the bow shock around the 4.5 mm tip radius.

## R.6 How far to trust the numbers (one table, citing the convergence section)

| Source | Size | Where |
|---|---|---|
| Iterative | C_D change < 0.1 % over the last 500 iterations, every run (M 0.98 and 1.00 needed one extra block) | `fluent_cd_<run>` |
| Residuals, pressure-based | 3.6-8 orders down; from M 1.05 the k residual creeps back up 1-2 orders while C_D stays flat | `fluent_residuals_<run>`, `runs/residuals.csv` |
| Residuals, density-based (M 1.20, burnout) | burnout >= 3 orders; M 1.20 continuity and energy stall at 2.2-2.4 orders (with the zigzag) | same |
| Balances, y+ | mass and heat < 1e-3 %; y+ < 1 on every wall (max 0.72) | `runs/summary.csv` |
| Mesh | GCI 1.08 % (A, M 0.95); 0.02 % at burnout (A and B); B at M 0.95 oscillatory, range 0.26 % | convergence section |
| Domain | 40 D within about 0.4 % at M 0.95; at M 1.00 not converged (several %); 20 D enough at M 1.20 (-0.07 % on 40 D) | convergence section |
| Solver switch at M 1.20 | pressure- vs density-based: A -0.17 %; B [to fill in] | R.5 |
| Model | stagnation pressure +0.09 to +0.90 % above theory (pressure-based, M 0.90-1.10), about 1e-4 in C_D; cone vs Taylor-Maccoll: surface Cp +1.3-1.5 %, shock angle within 0.2 deg | convergence section |

Main text: one pressure-based residual plot (e.g. `fluent_residuals_A_D40_M0.98`) and one density-based
(`fluent_residuals_A_medium_M1.20`), with their C_D histories; the rest in an appendix if the template allows.

## R.7 Summary for the conclusion

* Drag rise at M 0.950 (A) and 0.937 (B): B's drag rises earlier and more steeply.
* B has the lower drag over the flight except a narrow band M 0.958-1.019 (at most 6.3 % higher, at M 0.98);
  from M 1.05 to burnout it is 5-16 % lower, and 7 % lower in friction everywhere.

## Figure list

| Figure | Use |
|---|---|
| `fig_cd_mach.pdf` | R.1-R.4, the main figure |
| `fluent_mach_{A,B}_D40_M0.98`, `fluent_cp_AB_M0.98` | R.5, why B is higher at M 0.98 |
| `fluent_mach_wide_{A,B}_D40_M1.02` (or M 1.05) | R.5, detached bow shock |
| `fluent_cp_AB_burnout`, `fluent_mach_{A,B}_medium_burnout` | R.5, why B is lower supersonic |
| `fluent_cp_AB_M1.20_pressure`, `fluent_cp_solver_{A,B}_M1.20` | R.5, M 1.20 wall Cp and the zigzag |
| `fluent_cf_AB_<point>` | R.4, friction along the wall |
| `fluent_residuals_*`, `fluent_cd_*` | R.6 |
| `fluent_yplus_AB_<point>` | mesh section (y+ < 1) |

# Fluent study A: LV-Haack nose, 2D axisymmetric RANS

Verification runs of the report-2 design nose (`configs/weisshorn_650.yaml`): LV-Haack, curved
length 561 mm (L/D = 3.117) plus an 89 mm coupler, D = 180 mm, tip radius 4.5 mm, zero incidence.
Steady RANS (k-ω SST) in Ansys Fluent 2024 R2 at three points of the nominal Weisshorn trajectory,
following `CFD/2027_C_FD_REPORT-4_NOSE-CONE-FLUENT-WORKFLOW.pdf`. Run on 26-27 Oct 2026.

The full log with charts is [`studyA_log.html`](studyA_log.html) (download and open it in a browser).

## Results

Drag on `wall_nose` only (curved nose and blunt tip, up to the shoulder), A_ref = 0.02545 m².

| | M = 0.95 | M = 1.20 | M = 1.47 |
|---|---:|---:|---:|
| Altitude | 928 m | 1478 m | 2189 m |
| Pressure C_D | 0.0042 | 0.0948 | 0.0963 |
| Viscous C_D | 0.0248 | 0.0224 | 0.0207 |
| **Total C_D** | **0.0291** | **0.1172** | **0.1170** |
| Drag force | 42.3 N | 254.8 N | 349.4 N |
| Max y+ nose / body | 0.51 / 0.28 | 0.60 / 0.30 | 0.71 / 0.29 |
| C_D change over the last 500 iterations | 0.009 % | 0.083 % | 0.032 % |
| Iterations | 1200 | 3200 | 3000 |
| Solver | pressure-based coupled | density-based | density-based |

Above M = 1 the nose drag is mostly wave drag, about 0.095 and nearly flat between M = 1.2 and
1.47. At M = 0.95 the pressure drag almost disappears and friction is 85 % of the total. The
pressure parts agree within ±4 % with the Euler study `f650` interpolated to L/D = 3.117 (below).

## Contents

| Path | Contents |
|---|---|
| `studyA_log.html` | Full log: geometry, mesh, formulas, solver, runs, checks, problems and fixes |
| `freestream_bc.py` | Recomputes every free-stream and boundary value and the C_D from the wall forces |
| `wall_data/` | C_p, C_f, y+, pressure and wall shear on each `wall_nose` face of the final solutions (CSV, Fluent export, not sorted by x) |
| `cd_history/` | C_D on `wall_nose` per iteration for each run (CSV) |
| `plots/` | C_D, residual and Mach-contour pictures from Fluent |
| `../geometry_650/studyA_domain2D_2024R2.dsco` | 2D flow domain (Discovery 2024 R2) |

The Fluent case and data files (20-24 MB each) are not in git (`CFD/runs/` is ignored). They are
`studyA_M1.47`, `studyA_M1.2`, `studyA_M0.95` and `studyA_M0.95_unconverged` (`.cas.h5` and
`.dat.h5`) in `CFD/runs/` on the DLL VM, backed up in `Z:\weisshorn_studyA_2026-10-26\`.

## 1 Geometry

* CAD: `CFD/geometry_650/nose_lvhaack_L561_c89_rn4.5mm.step`, solid of revolution in mm, x axis =
  rocket axis, tip at the origin. Shoulder at x = 558.65 mm, r = 90 mm (the 4.5 mm tip blunting
  shortens the nominal 561 mm by 2.35 mm); coupler end at x = 647.65 mm.
* Workbench: Fluid Flow (Fluent) system, Geometry → Analysis Type = 2D before importing.
* Discovery: pull the coupler end disc +1800 mm to x = 2447.65 mm; sketch the rectangle
  (−3600, 0) to (2447.65, 3600) mm in the XY plane; Combine → Subtract from target (rectangle
  target, solid cutter); delete the solid and the inner piece. One surface remains, y ≥ 0.
* Why: zero incidence makes the flow axisymmetric, and Fluent integrates forces over 360°. Far
  field 20D upstream and radially; outlet 10.5D behind the shoulder. The cylinder runs out of the
  domain, so there is no base wake, as in the team's Euler nose-plus-cylinder setup.

| Named selection | Edge [m] |
|---|---|
| `inlet` | x = −3.6, y 0 → 3.6 |
| `farfield` | y = 3.6, x −3.6 → 2.44765 |
| `outlet` | x = 2.44765, y 0.09 → 3.6 |
| `axis` | y = 0, x −3.6 → 0 |
| `wall_nose` | curved nose, x 0 → 0.55865 |
| `wall_body` | cylinder, y = 0.09, x 0.55865 → 2.44765 |

## 2 Mesh (Ansys Meshing 2024 R2)

* Defaults: CFD / Fluent, linear, element size 0.05 m, growth 1.1, smoothing high.
* Quadrilateral inflation layers, triangles elsewhere; 169 674 cells, 143 250 nodes.
* Edge sizing 1 mm on `wall_nose`; 1 mm with bias factor 3 (small cells at the shoulder) on
  `wall_body`. Bias 10 gave 38 cells with skewness 0.985 near the outlet.
* Tip: sphere of influence at the origin, radius 15 mm, 0.2 mm cells (about 35 cells around the
  quarter-circle tip, π r_n / 2 = 7.07 mm).
* Inflation on both walls: first layer 1 µm, 45 layers, growth 1.2.
* Quality: max skewness 0.92, min orthogonal quality 0.237 (Fluent's metric 0.077); max aspect
  ratio 1660 in the first layer at the shoulder.

First cell for y+ = 1 (turbulent flat plate at the shoulder, x = 0.559 m):

```
Re_x = ρ V x / μ        C_f ≈ 0.026 Re_x^(-1/7)        u_τ = √(½ C_f V²)        y₁ = μ / (ρ u_τ)
→ 1.02 µm (M 1.47), 1.16 µm (M 1.2), 1.38 µm (M 0.95)   →  1 µm used, measured max y+ 0.51-0.71
```

Inflation thickness and boundary layer:

```
H = h₁ (r^N − 1)/(r − 1) = 1 µm × (1.2^45 − 1)/0.2 = 18.3 mm     last layer = 1 µm × 1.2^44 = 3.05 mm
δ ≈ 0.37 x Re_x^(-1/5) ≈ 7.5-8.0 mm at the shoulder               → the stack contains the boundary layer
```

## 3 Free stream and boundary conditions

Inputs from the trajectory table of the workflow note: M, static p, static T. Air is Fluent's
default ideal gas:

```
R = 8314.46 / 28.966 = 287.04 J/(kg K)       γ = c_p/(c_p − R) = 1006.43/719.39 = 1.399
μ = 1.716e-5 (T/273.11)^1.5 (273.11 + 110.56)/(T + 110.56)          (Sutherland, 3 coefficients)
ρ = p/(R T)    a = √(γ R T)    V = M a    q = ½ ρ V² = ½ γ p M²    Re_D = ρ V D / μ
T₀ = T (1 + (γ−1)/2 M²)    p₀ = p (1 + (γ−1)/2 M²)^(γ/(γ−1))
```

| | M = 0.95 | M = 1.20 | M = 1.47 |
|---|---:|---:|---:|
| p (input) | 90 666 Pa | 84 789 Pa | 77 649 Pa |
| T (input) | 282.1 K | 278.5 K | 273.9 K |
| ρ | 1.1197 kg/m³ | 1.0606 kg/m³ | 0.9876 kg/m³ |
| a | 336.6 m/s | 334.4 m/s | 331.7 m/s |
| V | 319.7 m/s | 401.3 m/s | 487.5 m/s |
| μ | 1.760e-5 Pa s | 1.743e-5 Pa s | 1.720e-5 Pa s |
| q | 57 238 Pa | 85 407 Pa | 117 371 Pa |
| Re_D | 3.66e6 | 4.40e6 | 5.04e6 |
| T₀ / p₀ | 332.9 K / 162 012 Pa | 358.5 K / 205 534 Pa | 392.0 K / 272 871 Pa |

| Zone | Type | Values |
|---|---|---|
| `inlet`, `farfield` | Pressure far-field | gauge pressure p, Mach M, temperature T, direction (1, 0) |
| `outlet` | Pressure outlet | gauge pressure p, backflow total temperature T, normal backflow, no averaging or target mass flow |
| `axis` | Axis | |
| `wall_nose`, `wall_body` | Wall | no-slip, stationary, heat flux 0 (adiabatic), smooth |

Operating pressure 0 Pa, so all pressures are absolute.

Turbulence on `inlet`, `farfield` and `outlet`: intensity I = 1 %, viscosity ratio μ_t/μ = 10,
chosen for calm-air external flow (not prescribed by the workflow). Fluent converts them to

```
k = 3/2 (I V)²        ω = ρ k / (μ · μ_t/μ)
M 0.95: k = 15.3 m²/s², ω = 9.76e4 1/s     M 1.2: k = 24.2, ω = 1.47e5     M 1.47: k = 35.7, ω = 2.05e5
```

Fluent's defaults for a new boundary (k = 1, ω = 1) give μ_t ≈ 1 Pa s, about 60 000 times the
laminar viscosity; they were replaced before the first run.

Reference values: A_ref = π D²/4 = 0.025447 m² (entered as 0.02545), length 0.18 m (moments
only), ρ, V, p, T computed from `inlet` for each Mach number. C_D = F_x / (q A_ref).

## 4 Solver

* Common: 2D double precision, axisymmetric, steady; k-ω SST (compressibility effects off);
  energy on; ideal gas with Sutherland viscosity; hybrid initialisation; drag report `cd-nose` on
  `wall_nose`, direction (1, 0).
* M = 1.47 and 1.2: density-based implicit, Roe-FDS, least-squares gradients, first-order upwind
  then second-order upwind.
* M = 0.95: pressure-based, Coupled with Fluent's pseudo-transient method, pressure "Second
  Order", second-order upwind for the rest after 200 first-order iterations, viscous heating on,
  residual stopping criteria set to 1e-12.
* Convergence: (max C_D − min C_D) / mean C_D over the last 500 iterations < 0.1 %. Residuals
  alone do not count.

## 5 Runs

| Mach | Iterations | Order | Courant | Note |
|---|---|---|---:|---|
| 1.47 | 1-100 | first | 1 | |
| | 101-400 | first | 5 | |
| | 401-3000 | second | 5 | Δ = 0.032 % over 2501-3000 |
| 1.2 | 1-100 | first | 1 | fresh initialisation (restart from M 1.47 went unstable) |
| | 101-400 | first | 5 | |
| | 401-2400 | second | 5 | residuals rose 1.3e-3 → 8e-3 |
| | 2401-3200 | second | 3 | residuals flat; Δ = 0.083 % over 2701-3200 |
| 0.95 | 1-200 | first | pseudo-transient | pressure-based |
| | 201-1200 | second | pseudo-transient | Δ = 0.009 % over 701-1200 |

At M = 0.95 the density-based solver was tried first and never settled: C_D cycled between 0.009
and 0.053, between 0.030 and 0.040 with a non-reflecting outlet, and between 0.016 and 0.031 at
Courant 1 (`cd_history/M0.95_density-based_not-converged.csv`). The pressure-based coupled solver
converged without oscillation.

## 6 Checks

* C_D from force: 349.42 N / (117 371 Pa × 0.02545 m²) = 0.1170; 254.75 / (85 407 × 0.02545) =
  0.1172; 42.32 / (57 238 × 0.02545) = 0.0291.
* Euler study `f650` (`data/euler_results.csv`, LV-Haack, sharp tip) at L/D 2.5 and 3.5,
  interpolated to 3.117 linearly in L/D and in 1/(L/D)²:

  | Mach | Euler 2.5 | Euler 3.5 | Interpolated | Fluent pressure C_D |
  |---|---:|---:|---:|---:|
  | 1.2 | 0.1302 | 0.0767 | 0.091-0.097 | 0.0948 |
  | 1.47 | 0.1356 / 0.1347 (M 1.4 / 1.6) | 0.0776 / 0.0766 | 0.093-0.100 | 0.0963 |

* Viscous C_D on `wall_body`: 0.0745 / 0.0824 / 0.0835 (M 1.47 / 1.2 / 0.95), consistent.

## 7 Open items

* M = 0.95 used a different solver from the other two; re-running M = 1.2 with the pressure-based
  solver would check that the choice does not shift the result.
* The outlet backflow total temperature was set to the static T, not T₀. It only acts on reversed
  flow at the outlet; state it in the report or repeat with T₀.
* Mesh study at M = 1.47 (coarse, medium = this mesh, fine; about ×1.5 per direction).
* Turbulence boundary sensitivity (1 % / 10 against 0.1 % / 1).
* 3D half-model at 5° incidence for C_Nα and the centre of pressure.

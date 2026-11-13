# ME-474 Assessment 2, group 13: Fluent runs of the two 720 mm noses

2D axisymmetric steady RANS (k-ω SST) of nose A (LV-Haack 679 + 41 mm) and nose B (Von Kármán
700 + 20 mm), D = 180 mm, tip radius 4.5 mm, Ansys Fluent 2024 R2. The setup is a copy of the Rocket
Team study `CFD/study720`, with the fixes below. Due 1 January 2027, one PDF per group.

## Contents

| File | What it does |
|---|---|
| `make_points.py` | Free stream at M 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20 and burnout on design A's simulated ascent: `python ME474/make_points.py` |
| `fluent_points.csv` | Its output: p, T, ρ, V, q, Re_L, T0, p0, pitot p0, k, ω per point |
| `setup_fluent.py` | Fluent setup: models, boundary conditions, reports, solver schedules, exports |
| `run_queue.py` | Runs cases one after the other: `python run_queue.py MESH@POINT[:SOLVER] ...` |
| `check_run.py` | Convergence and sanity checks of every run → `runs/summary.csv` (no Fluent needed) |
| `gci.py` | Grid Convergence Index from three meshes (Celik et al. 2008) |
| `meshes/` | The new meshes (`A_coarse.msh`, `A_medium.msh`, ...). `A_study720` and `B_study720` are the Rocket Team meshes in `CFD/study720/mesh` (174k cells), used where they are |
| `runs/<run>/` | Everything a run writes. Git keeps the text outputs and ignores the case/data files (`*.h5`) and `autosave/` |

## Changes from study720

1. **Backflow total temperature = free-stream T0** (was the static T∞) on the pressure outlet.
2. **Mass and heat flux reports** on `inlet`, `farfield`, `outlet` (and the walls for heat), plus their
   net sums, logged every iteration to `<run>_flux.out`. Week 8 asks for both balances within 0.2 %.
3. **More output after convergence:**
   - `<run>_axis.csv`: total pressure and Mach along the axis, i.e. the stagnation streamline from the
     inlet to the tip.
   - `<run>_report.txt`: wall forces (pressure and viscous C_D), `mesh/check` and `mesh/quality`.
4. **Parametric runs:** any mesh in `meshes/` at any point of `fluent_points.csv`, with all files in
   `runs/<run>/`. The 300 K wall stage is dropped, since the course study keeps the walls adiabatic.

Items 2 and 3 have **not run in Fluent yet**. The first run is a smoke test: check that
`<run>_flux.out`, `<run>_axis.csv` and `<run>_report.txt` appear and that `check_run.py` fills every
column. If the flux report keys are refused, `flux_reports` prints the allowed values.

## Running on the VM

1. Pull the repository on the VM (`CFD/mcp/bootstrap-vm.ps1` starts Claude Code in it).
2. Start Fluent **2024 R2** (not 2026 R1) in 2D, double precision. In its console:
   `(chdir "<repo>/ME474")`, `define/beta-feature-access` (yes), `server/start-server`. Fluent writes
   `server_info-<pid>.txt` here; the scripts connect to the newest one (git ignores them).
3. Run cases, then summarise them:
   ```
   cd ME474
   python run_queue.py A_study720@M0.95 A_study720@M1.20:pressure
   python check_run.py
   ```
   A run takes hours: start `run_queue.py` in the background with its output in a log file.
4. Commit the text outputs of `runs/` to get them back on your PC. The case/data files stay on the VM.

The solver defaults to pressure-based Coupled up to M 1.10 and density-based Roe-FDS from M 1.20.
`:pressure` or `:density` overrides it, and the run name then gets that suffix. Use `finish:RUN` to
continue the case already in the session until it converges.

## Free stream (design A's ascent, used for both shapes)

| Point | z [m] | M | p [Pa] | T [K] | T0 [K] | p0 [Pa] | pitot p0 [Pa] |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0.95 | 923 | 0.95 | 90714.2 | 282.149 | 332.951 | 162098 | 162098 |
| M0.98 | 982 | 0.98 | 90066.7 | 281.765 | 335.752 | 166532 | 166532 |
| M1.00 | 1023 | 1.00 | 89623.2 | 281.500 | 337.660 | 169597 | 169597 |
| M1.02 | 1065 | 1.02 | 89170.6 | 281.229 | 339.602 | 172749 | 172748 |
| M1.05 | 1129 | 1.05 | 88478.0 | 280.812 | 342.578 | 177653 | 177627 |
| M1.10 | 1240 | 1.10 | 87290.7 | 280.091 | 347.705 | 186313 | 186113 |
| M1.20 | 1476 | 1.20 | 84801.4 | 278.554 | 358.579 | 205564 | 204083 |
| burnout | 2185 | 1.46548 | 77684.7 | 273.947 | 391.323 | 271231 | 255044 |

M 0.95, 1.20 and burnout are the study720 values. M 0.98 and 1.02 are the fall-backs if M 1.00
does not converge. "pitot p0" is the total pressure behind a normal shock: the highest wall pressure
of a converged run cannot exceed it.

## Checks per run (`check_run.py`)

| Column | Target |
|---|---|
| `cd_spread_pct` | < 0.1 % over the last 500 iterations (the run stops on this) |
| `mass_imb_pct`, `heat_imb_pct` | < 0.2 % of the inlet flux (week 8). The inlet heat flux is Fluent's total enthalpy relative to its 298.15 K reference, which makes this check stricter |
| `p_stag_pct` | ≤ 0. The study720 runs at M 0.95 gave +1.23 % (A) and +0.93 % (B), with the pressure-based solver only |
| `p0_inlet_pct`, `p0_axis_max_pct` | ≈ 0. Shows where along the stagnation streamline the extra total pressure appears |
| `yplus_max` | < 1 |
| `min_orth_q` | study720 had 0.062 to 0.088 at the x = 720 mm wall-size step; aim for > 0.15 |

Residuals are not a stop criterion. Report them from the Fluent plots: a drop of 3 orders, or an
explanation where they stall. At burnout, continuity only drops about 2.5 orders.

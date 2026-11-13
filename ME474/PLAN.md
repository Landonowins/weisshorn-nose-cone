# ME-474 run plan, group 13 (draft for approval)

Written on 13 Nov 2026 on the DLL VM while the smoke tests ran. Nothing long starts before the
plan is approved. The decision points (section 7) are for you to settle. Every time below scales
from one measurement: the pressure-based speed on the old 174k mesh.

## 1. Environment (DLL VM DLLSTIGEPC211, 13 Nov 2026)

| Item | Found |
|---|---|
| Fluent | 2024 R2 (build 10192, 13 May 2024), 2D double precision, academic licence from `1055@software.epfl.ch` |
| Cores | i7-11700: 8 physical cores, 16 threads, 64 GB RAM. Fluent runs 8 processes (n0-n7), one per core |
| Licences | Not a limit: `cfd_solve_level1/2` 1650 seats each (15 in use), `anshpc` 2700 (568 in use). This session holds 4 `anshpc` for its 8 processes. The limit is the 8 cores per VM |
| Disk | `C:` 550 GB free (purged with the profile). `Z:` = `\\files9\data\feig` (personal EPFL home) 2.3 GB free of 5 GB. `M:` = `\\sti-nas1.rcp.epfl.ch\MAKE` 67.7 GB free of 2 TB (has `rocket_team\`) |
| PyFluent | 0.43.0 on Python 3.14.7, `C:\Program Files\Python314\python.exe` (mcp 2.3.0). scipy is not installed, so `nosecone/taylor_maccoll.py` does not run here (the cone targets below come from your PC) |
| Session limits | Group policy (`HKLM\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services`): `MaxIdleTime` = 12 h (an idle session is **disconnected**; `fResetBroken` is not set, so it is not ended), `MaxDisconnectionTime` = 24 h (a disconnected session is **logged off**: Fluent dies and C: may be purged), no `MaxConnectionTime` (no total limit). So it is not "2 h idle / 24 h total": a queue survives **24 h after you close RDP** (or 12 + 24 h with RDP open and idle). Reconnect at least once a day while runs go on |
| Last boot | 13 Nov 2026 19:03 |

Speed (pressure-based Coupled, A_study720, 174 380 cells, 8 processes): **0.45 s/iteration first
order, 0.39 s/iteration second order** (iterations 312 → 496 in 72 s). Density-based Roe-FDS has not
run yet on this VM: below I assume **0.50 s/iteration** at 174k and correct it after the first density run.
Time per iteration scales with the cell count.

## 2. Smoke tests

**Dry run first** (20 iterations of A_study720 at M 0.95, outputs in the scratchpad, not in `runs/`):
every new Fluent call was accepted (flux report definitions, axis export, wall export, transcript
with wall forces, `mesh/check`, `mesh/quality`). It found one bug, which is now fixed:

* `check_run.read_report()` read the column names from the first header line of a Fluent `.out`
  file, `"Iteration" "cd-total etc.."`, which names only one column. Fluent 2024 R2 writes the full list on the next line,
  `("Iteration" "cd-total" "cd-nose")`. `run_queue.converge()` uses the same function, so the
  first real run would have crashed with `KeyError: 'cd-nose'` right after its 1200-iteration
  schedule: no exports, and the second run would never start. The fix takes the longest list of names.
  After it, `check_run.py` fills all 20 columns from the dry-run files.

**Queue** `python run_queue.py A_study720@M0.95 A_study720@M1.20:pressure` (log `runs/queue_smoke.txt`):
both runs finished without errors, 8 min each (22:34 to 22:42 and 22:42 to 22:50), including
setup and exports. `check_run.py` fills every column of `runs/summary.csv`.

### 2a. Smoke results (old mesh A_study720, 174 380 cells, pressure-based Coupled)

| | A_study720_M0.95 | A_study720_M1.20_pressure | study720 reference |
|---|---:|---:|---|
| Iterations, cd-nose spread over the last 500 | 1200, 0.009 % | 1200, 0.000 % | |
| `wall_nose` C_D (pressure + viscous) | **0.03200** (0.00317 + 0.02883) | **0.09456** (0.06832 + 0.02623) | M 0.95: 0.03200 (0.0032 + 0.0288). M 1.20, density-based: 0.0949 (0.0686 + 0.0263) |
| Mass / heat imbalance (net / inlet) | 7e-8 % / 2e-7 % | 5e-9 % / 6e-9 % | target < 0.2 % |
| Residual drop, iteration 1 → 1200 | continuity 6.0 orders, the others 5.6 to 7.7 | continuity 5.7, k 3.3, ω 3.7 orders | k and ω stall near 1e-4 in second order at M 1.20 |
| Total pressure: inlet / axis max | 0.000 % / +1.445 % (x = −6.9 mm) | 0.000 % / +2.05 % (inside the shock, x = −6.9 mm) | |
| Highest wall pressure vs theoretical stagnation pressure | **+1.231 %** | **−0.006 %** of pitot | M 0.95 pressure-based: +1.23 %. M 1.20 density-based: −0.3 to −0.5 % |
| Max y+ | 0.54 | 0.63 | |
| Min orthogonal quality / max aspect ratio | 0.088 at (718.5, 90.0) mm / 1648 at the outlet end of `wall_body` | same mesh | |

**Backflow T0 fix:** no visible effect. C_D, its split and the stagnation excess are the same as in study720
to all printed digits. With the body running out through the outlet, there is probably no backflow at all.

**Axis diagnosis (item 6), M 0.95:** the inlet is clean (p0 −0.0001 %, T0 exact), so the boundary condition
and the domain are not the cause. The total pressure then rises gradually along the decelerating
stagnation streamline, while T0 stays within 0.002 % (an entropy decrease, not an energy error):

| x [mm] | −978 | −481 | −201 | −105 | −49 | −20 | −10 | −2.1 | −0.2 | 0 (wall) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mach | 0.949 | 0.945 | 0.935 | 0.917 | 0.885 | 0.822 | 0.742 | 0.431 | 0.073 | 0.0005 |
| p0 / p0∞ − 1 [%] | +0.003 | +0.020 | +0.088 | +0.228 | +0.487 | +0.926 | +1.168 | +1.228 | +1.261 | +1.231 |

It is neither "high at the inlet" (boundary condition) nor "only at the tip": it builds up wherever
the flow slows down in growing cells, mostly in the last 100 mm before the tip. That is outside the
old 15 mm, 0.2 mm sphere, where the cells grow towards 50 mm. On top of that, one axis face at x = −6.9 mm
has +1.445 %, a local spike, probably the cell at the top of the inflation stack in front of the tip.

**Does it follow the pressure-based solver? No.** At M 1.20 the same solver gives a stagnation pressure
within −0.006 % of pitot. Behind the bow shock (standoff 7 mm) the axis total pressure is −0.66 to −0.73 %,
against −0.72 % for a normal shock. There the deceleration happens in the 7 mm shock layer, inside
the 0.2 mm cells. At M 0.95 it happens over about a metre of coarse cells. The +2.05 % axis maximum
at M 1.20 lies inside the captured shock, where a total pressure computed from smeared p and M has no meaning.
So the excess most likely follows the mesh along the subsonic deceleration, not the solver. Two checks:

* the mesh study at M 0.95 (group 1): `p_stag_pct` and `p0_axis_max_pct` should fall from A_coarse to
  A_fine, and the 40 mm sphere already puts the last 40 mm before the tip at 0.2 mm;
* recommended extra run on the old mesh, 0.6 h: `A_study720@M0.95:density`. If the density-based solver
  also gives about +1.2 %, the cause is the mesh. If not, it is the pressure-based discretisation.

**Solver cross-check on the old mesh, M 1.20:** pressure-based C_D 0.09456 against density-based 0.0949
(study720): −0.4 % in total, −0.4 % pressure and −0.3 % viscous. The pressure-based solver is fit for the
transonic points. Group 4 repeats this on A_medium.

## 3. Run list, in order

Syntax of `run_queue.py`. Pressure-based up to M 1.10, density-based from M 1.20 (the defaults).

| # | Needs | Purpose | Queue |
|---|---|---|---|
| 0 | old mesh | smoke (done, section 2a) | `A_study720@M0.95 A_study720@M1.20:pressure` |
| 0b | old mesh | recommended: is the M 0.95 excess the mesh or the solver? (0.6 h, can run now) | `A_study720@M0.95:density` |
| 1 | A_coarse, A_medium, A_fine | mesh study, subsonic | `A_coarse@M0.95 A_medium@M0.95 A_fine@M0.95` |
| 2 | A_coarse, A_medium, A_fine | mesh study, burnout | `A_coarse@burnout A_medium@burnout A_fine@burnout` |
| 3 | A_D10, A_D40 (+ A_medium from 1) | domain study | `A_D10@M0.95 A_D40@M0.95` |
| 4 | A_medium | solver cross-check (the density run is also production) | `A_medium@M1.20:pressure A_medium@M1.20` |
| 5 | cone | validation (Taylor-Maccoll) | `cone@burnout cone@M1.20` |
| 6 | A_medium | production A (M 0.95 and burnout come from 1 and 2) | `A_medium@M0.98 A_medium@M1.00 A_medium@M1.05 A_medium@M1.10` |
| 7 | B_medium | production B | `B_medium@M0.95 B_medium@M0.98 B_medium@M1.00 B_medium@M1.05 B_medium@M1.10 B_medium@M1.20 B_medium@burnout` |
| 8 | (fallback) | only if M 1.00 does not converge | `A_medium@M1.02 B_medium@M1.02` |
| 9 | (optional) | needs a new script option, which I would ask about first | first order to convergence `A_medium@burnout`; turbulence inlet 0.1 %/1 vs 1 %/10 |

* M 0.98 is in production, not only a fallback: the drag-rise definition needs the slope at the
  start of the rise (decision 3).
* Production assumes the medium mesh and the 20 D domain. If decision 1 picks the fine mesh,
  groups 6 and 7 move to `A_fine` / `B_fine` (B_fine still to mesh) and cost about twice as much.
  If decision 2 picks 40 D, the points up to M 1.10 move to `A_D40` / `B_D40`.
* After every run: `python check_run.py`, copy the case/data (section 6), commit and push `ME474/runs/`.

Start long queues detached, so that they survive a restart of Claude Code (PowerShell):

```powershell
cd C:\Users\feig\weisshorn-nose-cone-git\ME474
Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Python314\python.exe" `
  -ArgumentList "-u run_queue.py A_coarse@M0.95 A_medium@M0.95 A_fine@M0.95" `
  -RedirectStandardOutput runs\queue_1.txt -RedirectStandardError runs\queue_1.err.txt
```

The log holds Fluent's residual table for every iteration, so the residual plots can come from
the log (it is committed) as well as from `fluent_figs.py` on the saved case/data.

## 4. Estimated time

Cells are estimated (section 5). "Schedule" = 1200 iterations (pressure) or 4000 (density), the length at which every
study720 run converged. "Max" = schedule + 8 blocks of 500, the most `run_queue.py` runs before it
reports NOT CONVERGED. About 3 min per run for reading the mesh, initialising and exporting comes on top.

| Run | Cells (est.) | Solver | s/iter (est.) | Schedule [h] | Max [h] |
|---|---:|---|---:|---:|---:|
| A_study720 (smoke, ×2) | 174k | pressure | 0.39 | 0.13 each | 0.56 each |
| A_coarse@M0.95 | 130k | pressure | 0.29 | 0.10 | 0.42 |
| A_medium@M0.95 (and every medium pressure run, A or B) | 250k | pressure | 0.56 | 0.19 | 0.81 |
| A_fine@M0.95 | 480k | pressure | 1.08 | 0.36 | 1.56 |
| A_coarse@burnout | 130k | density | 0.37 | 0.41 | 0.82 |
| A_medium@burnout (and every medium density run) | 250k | density | 0.72 | 0.80 | 1.60 |
| A_fine@burnout | 480k | density | 1.38 | 1.53 | 3.07 |
| A_D10@M0.95 | 237k | pressure | 0.53 | 0.18 | 0.77 |
| A_D40@M0.95 | 294k | pressure | 0.66 | 0.22 | 0.95 |
| cone@burnout, cone@M1.20 | 265k | density | 0.76 | 0.85 each | 1.69 each |

| Group | Schedule [h] | Max [h] |
|---|---:|---:|
| 1 + 2 mesh study | 3.4 | 8.3 |
| 3 domain study | 0.4 | 1.7 |
| 4 + 5 solver check and cone | 2.7 | 5.8 |
| 6 + 7 production (medium) | 3.3 | 10.5 |
| **Total, one VM** | **≈ 10** | **≈ 26** |
| 8 fallbacks | 0.4 | 1.6 |

The transonic points (M 0.98 to 1.10) are the most likely to need the extra blocks.

**On several VMs** (each with its own clone on C:, Fluent 2024 R2 with 8 processes started in `ME474/`,
4 `anshpc` each):

* VM 1: groups 1 + 2 (mesh study), 3.4 to 8.3 h.
* VM 2: groups 3 + 4 + 5 (domain, solver check, cone), 3.1 to 7.5 h.
* VM 3: groups 6 + 7 once decisions 1 to 3 are settled (3.3 to 10.5 h). Or start group 7 right away on
  B_medium, at the risk of redoing it if the GCI picks the fine mesh.

Each run writes only to its own `runs/<run>/`, so commits from several VMs do not conflict.
The exception is `runs/summary.csv`: after `git pull --rebase`, run `python check_run.py` again
and commit that summary. Commit the new meshes (`meshes/*.msh`, 5 to 40 MB, under GitHub's 100 MB limit) so that every VM gets them and they survive a purge.
On another VM, start Fluent in `ME474/`, or type the Scheme form `(chdir "C:/Users/feig/weisshorn-nose-cone-git/ME474")` with its parentheses: today's
`chdir "C:\...` without them failed, which was harmless here because Fluent had started in ME474.

## 5. Mesh settings (Workbench Meshing, by hand)

Recipe of study720 with the two changes of the plan (tip sphere 40 mm, no wall-size step at
x = 720 mm), then every size × 1.5 between levels. Inflation stays the same on all levels.

| Setting | A_coarse | A_medium (= B_medium, A_D10, A_D40, cone) | A_fine |
|---|---:|---:|---:|
| Method | triangles, as study720 | triangles | triangles |
| Global element size | 75 mm | 50 mm | 33.3 mm |
| Edge sizing `wall_nose` + `wall_coupler`, element size | 1.5 mm | 1.0 mm | 0.667 mm |
| Edge sizing `wall_body`: number of divisions, bias factor 3, small end at the coupler | 660 | 989 | 1484 |
| → `wall_body` first / last cell | 1.5 / 4.5 mm | 1.0 / 3.0 mm | 0.667 / 2.0 mm |
| Sphere of influence at the tip (centre x = 0, y = 0), radius 40 mm, element size | 0.30 mm | 0.20 mm | 0.133 mm |
| Inflation on the three walls: first layer, layers, growth | 1 µm, 45, 1.2 | 1 µm, 45, 1.2 | 1 µm, 45, 1.2 |
| Expected cells (estimate) | ≈ 130k | ≈ 250k | ≈ 480k |

* Behaviour (hard/soft), growth rate and the other defaults: as in study720.
* `wall_body` divisions: for a 1800 mm body (outlet at 2520 mm) with bias 3. The bias factor is the
  largest over the smallest edge, and the first cell equals the coupler's size, so there is no step at x = 720 mm.
  In the old mesh, "1 mm, bias 3" over 1800 divisions gave 0.549 mm at x = 720, which was the
  1 → 0.55 mm step. For the cone with its outlet at 2520 mm (1780 mm of body) the medium count is 978.
* Inflation: 45 layers of growth 1.2 from 1 µm reach 18.3 mm, with a last layer of 3.05 mm.
* Aspect ratio: the first layer (1 µm) on the last `wall_body` cell gives 4500 / 3000 / 2000 at the outlet
  end (coarse / medium / fine), against 1648 in the old mesh, with which both solvers converged. It is far behind the
  nose. If a coarse run converges badly, look there first; otherwise it only explains `max_aspect` in `summary.csv`.
* **Cell count:** the old A mesh has 174 380 cells. Of these, 45 × 2599 wall faces = 117k are inflation quads
  (wall_nose 758, wall_coupler 41, wall_body 1800 faces) and 57k are triangles. Inside a 40 mm radius around the
  tip there are 2100 mm² of fluid, which takes up to 121k triangles at 0.2 mm (54k at 0.3 mm, 273k at 0.133 mm),
  against 16k for the old 15 mm sphere. Half of A_medium is therefore in the tip sphere, and the counts are
  ≈ 130k / 250k / 480k, not 77k / 174k / 390k. Read the count of A_medium in Workbench before meshing
  the others. If it is much higher than this, tell me before the long runs.
* Refinement ratio for the GCI (gci.py: r = √(N_fine/N_coarse)): with these counts r21 ≈ 1.39 and
  r32 ≈ 1.36, above the 1.3 that Celik et al. ask for. N grows by less than 2.25 per level because the 45
  inflation layers are not refined. The study therefore refines along the wall and in the far field; the
  wall-normal spacing stays the same, with y+ < 1 on all three.
* Export: Fluent mesh to `ME474/meshes/<name>.msh` (names `A_coarse`, `A_medium`, `A_fine`, `B_medium`,
  `A_D10`, `A_D40`, `cone`).

## 6. Domains (built by you in Discovery)

Tip at x = 0, axis along x, mm. The outlet stays 10 D behind the coupler. Only the inlet distance
and the radial size change. The face area is the profile integral (as `make_domain_2d.py` checks
it), to compare with the area of your face. Study720 matched within 0.001 %.

| Domain | Inlet x | Far field y | Outlet x | Face area [mm²] |
|---|---:|---:|---:|---:|
| A_D10 | −1800 | 1800 | 2520 | 7.567314e+06 |
| A_medium, A_coarse, A_fine (20 D) | −3600 | 3600 | 2520 | 2.182331e+07 |
| A_D40 | −7200 | 7200 | 2520 | 6.977531e+07 |
| B_medium (20 D) | −3600 | 3600 | 2520 | 2.182686e+07 |
| cone (20 D) | −3600 | 3600 | 2520 or 2540 | 2.183760e+07 or 2.190780e+07 |

Cone outlet: your table gives 2520 mm. With the cone's coupler ending at 740 mm, "10 D behind
the coupler" is 2540 mm. Both are fine, since both cone runs are supersonic and the outlet cannot
reach back to the cone surface. 2540 keeps the 1800 mm body and the 989 divisions of A.

The seven named selections the scripts need (an edge each; the face itself can have any name):

| Named selection | A (LV-Haack 679 + 41) | B (Von Kármán 700 + 20) | cone (720 × 90, 7.125°) |
|---|---|---|---|
| `inlet` | x = inlet, y from 0 to the far field | same | same |
| `farfield` | y = far field, inlet to outlet | same | same |
| `outlet` | x = outlet, y from 90 to the far field | same | same |
| `axis` | y = 0, inlet to the tip (x = 0) | same | same |
| `wall_nose` | tip (0, 0) to the shoulder (679, 90) along `CFD/geometry_720/nose_lvhaack_L682_c41_rn4.5mm.csv` (arc 689.6 mm) | tip to (700, 90) along `nose_vonkarman_L705_c20_rn4.5mm.csv` (arc 709.2 mm) | straight line (0, 0) to (720, 90), sharp tip (725.6 mm) |
| `wall_coupler` | (679, 90) to (720, 90) | (700, 90) to (720, 90) | (720, 90) to (740, 90): needed, since the drag reports and the wall export look for it |
| `wall_body` | (720, 90) to the outlet | same | (740, 90) to the outlet |

Optional cross-check of your STEP: `python CFD/make_domain_2d.py CFD/geometry_720/nose_lvhaack_L682_c41_rn4.5mm.csv --coupler 41 --upstream 10 --radial 10 --downstream 10 --out <scratch>.step`
(`--upstream 40 --radial 40` for A_D40). I have not run it.

Cone targets (Taylor-Maccoll, from your PC): M 1.465: Cp 0.0723, shock 43.3°, surface M 1.392;
M 1.20: Cp 0.0859, shock 56.7°, surface M 1.135. Compare Cp along the cone away from the tip
and the shoulder, and read the shock angle from the Mach or pressure contour. The boundary-layer
displacement makes the effective cone slightly thicker, so a Cp up to about 1 % above the inviscid
value is expected.

## 7. Decision points

1. **Final mesh** (after groups 1 and 2): `python gci.py A_fine_M0.95 A_medium_M0.95 A_coarse_M0.95`,
   the same at burnout, each with `--field cd_nose`, `cd_p` and `cd_v`. Proposal: production on
   **medium** if convergence is monotonic and medium's distance to the Richardson value is smaller
   than half the A-B difference we want to resolve (13-15 % in study720 at M 1.2 and 1.47, so easy
   there). At M 0.95, where A ≈ B, the result is then "equal within the GCI". Otherwise fine.
2. **Final domain** (after group 3): proposal **20 D** if |C_D(20 D) − C_D(40 D)| is below the GCI
   of the medium mesh, otherwise 40 D for M ≤ 1.10 (B_D40 to mesh). At M 0.95 the 20 D domain acts like
   ≈ 6 D (√(1 − M²) = 0.31), and M 1.00 is the worst case. Optional, if 20 vs 40 D is close at M 0.95:
   `A_D40@M1.00` against `A_medium@M1.00` (0.2 to 0.8 h).
3. **Drag-rise Mach number**: proposal dC_D/dM = 0.1, from finite differences of the `wall_nose` C_D
   between neighbouring points (slope at each midpoint, crossing by linear interpolation). Study720 gives
   0.032 at M 0.95 and about 0.095 at M 1.20, a mean slope of 0.25, so the crossing lies between
   about 0.95 and 1.0. Hence M 0.98 in production. If the slope from 0.95 to 0.98 is already above 0.1,
   the crossing is below 0.95, and a point at M 0.90 has to be added to `fluent_points.csv` (via `make_points.py`;
   I would ask first).
4. **Where to keep the case/data** (section 8).
5. **Pressure-based solver for M ≤ 1.10**: depends on the smoke results (section 2a). It stays as it is
   unless they show a problem.

## 8. Case and data files of the final runs

Size: 14 MB per case + data at 174k cells (5 + 9 MB), so about 20 MB at 250k and 40 MB at 480k.
About 25 final runs need **≈ 0.6 GB**, without the `_init` files and `autosave/`.

Proposal: keep working on `C:` and copy each finished run's final `<run>.cas.h5` / `<run>.dat.h5` to your
personal EPFL home **`Z:\ME474\runs\`**. It holds 2.3 GB free, is backed up, is not purged with the VM
profile, and is visible from every DLL VM. AppLocker on Z: only blocks executables. Once per run, after `check_run.py`:

```powershell
robocopy C:\Users\feig\weisshorn-nose-cone-git\ME474\runs Z:\ME474\runs *.cas.h5 *.dat.h5 /S /XD autosave /XF *_init.cas.h5 *_init.dat.h5
```

`M:\rocket_team\` (67.7 GB free) can hold a second copy if the team agrees. It is a shared team
share, so I have not written there. `fluent_figs.py` then reads the cases from Z: on any DLL VM.

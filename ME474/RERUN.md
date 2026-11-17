# Rerunning the ME-474 study

Everything needed to redo the 39 Fluent runs of this study, and every number and figure made from them, is in this
repository. The runs need only the meshes (step 2) and the scripts; the geometry and the Workbench projects are here
to remake or change the meshes. Our own case and data files (`runs/*/*.h5`, 1.3 GB) are not in git; a copy is on the
DLL VM drive `Z:\ME474\runs`.

## Software

* Ansys 2024 R2: Fluent (2D runs), Workbench with Ansys Meshing, Discovery (geometry). The meshes and cases are
  2024 R2 files.
* Python 3.14 with PyFluent: `python -m pip install ansys-fluent-core==0.43.0` (we used 0.43.0 on Python 3.14.7).
  It runs `run_queue.py`, `setup_fluent.py`, `check_run.py`, `drag_curve.py`, `gci.py`, `residuals.py`,
  `fluent_figs.py`, `fluent_figs_study.py`, `shock_angle.py`.
* Ansys's own Python, `C:\Program Files\ANSYS\v242\commonfiles\CPython\3_10\winx64\Release\python\python.exe`:
  it has matplotlib, scipy and h5py, which the Python above lacks. It runs `report_figures.py`, `mesh_sizes.py`,
  `taylor_maccoll_targets.py`.
* Only to remake `fluent_points.csv` (it is committed): the repository's Python environment with the `nosecone`
  package (`make_points.py`).

## 1. Geometry

* Nose profiles r(x): `CFD/geometry_720/nose_lvhaack_L682_c41_rn4.5mm.csv` (A) and
  `nose_vonkarman_L705_c20_rn4.5mm.csv` (B), with their STEP files and equations in the same folder; B's STEP files
  `B_vonkarman_700_c20_rn4.5.step` (3D) and `B_vonkarman_700_c20_rn4.5_2d.step` (2D).
* 2D domains: `CFD/geometry_720/domain2d_*.step` (20 D); sizes of the 10 D, 20 D and 40 D domains and of the cone
  domain in PLAN.md section 6. `CFD/make_domain_2d.py` writes a domain STEP from a profile (optional cross-check).
* The Discovery geometry of every mesh is inside the Workbench project (step 2): `dp0/<system>/DM/*.dsco`.
  `workbench/B_2d_geometry_FFF-1.dsco` is B's 2D geometry, restored from the shared Drive on 14 Nov (PLAN.md 5).

## 2. Meshes

**To rerun the runs, use the committed meshes:** `meshes/*.msh` (A_coarse, A_medium, A_fine, B_coarse, B_medium,
B_fine, A_D10, A_D40, B_D40, cone) and `../CFD/study720/mesh/{A,B}_mesh.msh` (the Rocket Team meshes, runs
`A_study720_*`). Cells, nodes and faces: `runs/mesh_sizes.csv`; quality: `meshes/mesh_quality.csv` and PLAN.md 5.

To remake or change a mesh, open `workbench/me474_A_meshes.wbpj` in Workbench 2024 R2. The recipe (sizes, inflation,
tip sphere, named selections) and the settings of each mesh are in PLAN.md section 5; export each mesh as a Fluent
mesh to `meshes/<name>.msh`.

| Workbench system | Folder | Geometry | Meshes made from it |
|---|---|---|---|
| A | `FFF` | `FFF.dsco` | A_coarse, A_medium, A_fine (element sizes changed per PLAN.md 5; the last export is A_medium) |
| B | `FFF-1` | `FFF-1.agdb` (B's 2D geometry) | B_coarse, B_medium, B_fine (last export B_medium) |
| A_D10 | `FFF-2` | `FFF-2.dsco` | A_D10 |
| A_D40 | `FFF-3` | `FFF-3.dsco` | A_D40 (the export left in `FFF-3/MECH/FFF-3.msh` is an older one, equal to A_medium; A_D40.msh is in `meshes/`) |
| Cone | `FFF-4` | `FFF-4.dsco` | cone |
| B_D40 | `FFF-5` | `FFF-5.dsco` | B_D40 (duplicate of B's system) |

The project was saved on 17 Nov at 12:27 with all six systems and copied unchanged (without Workbench's lock and
cache files). The `.msh` exports inside it are byte-identical to the files in `meshes/` named in the table (except
`FFF-3.msh`, see above); A_coarse, A_fine, B_coarse and B_fine exist only in `meshes/`.

The Rocket Team's original project of the study720 meshes is `../CFD/study720/workbench/nose720_A_B.wbpj` (its B
export is byte-identical to `B_mesh.msh`).

## 3. Fluent session

The scripts drive a Fluent that you start yourself, with a server-info file in `ME474` (they connect to the newest
`server_info*.txt` there; it holds the session password, so it is git-ignored):

```powershell
cd <repository>\ME474
Start-Process -FilePath "C:\Program Files\ANSYS\v242\fluent\ntbin\win64\fluent.exe" `
  -ArgumentList "2ddp -t8 -sifile=server_info-A.txt" -WorkingDirectory (Get-Location)
```

2D, double precision, 8 processes. On the DLL VMs: disconnect from RDP but never sign out, or Fluent is killed.

## 4. The runs

`python run_queue.py SPEC ...` runs the specs one after the other. Each run reads its mesh, applies
`setup_fluent.py` (models, boundary conditions at the free stream of `fluent_points.csv`, reports, solver schedule),
iterates until the nose C_D changes by less than 0.1 % over 500 iterations, saves and exports to `runs/<run>/`
(README.md). Nothing is set by hand. SPEC = `MESH@POINT[:SOLVER]`; the solver is pressure-based up to M 1.10 and
density-based from M 1.20 unless given.

The 39 runs, in the groups of PLAN.md section 3 (each `run.json` gives the same mesh, point and solver):

| Group | Purpose | Specs |
|---|---|---|
| 0 | smoke test on the Rocket Team meshes (PLAN.md 2a) | `A_study720@M0.95 A_study720@M1.20:pressure` |
| 0b | density-based at M 0.95 on the old mesh (2b: limit cycle) | `A_study720@M0.95:density` |
| 1 | mesh study A, M 0.95 (20 D) | `A_coarse@M0.95 A_medium@M0.95 A_fine@M0.95` |
| 2 | mesh study A, burnout (20 D) | `A_coarse@burnout A_medium@burnout A_fine@burnout` |
| 2B | mesh study B, M 0.95 and burnout; B_medium@burnout is also production | `B_coarse@M0.95 B_medium@M0.95 B_fine@M0.95 B_coarse@burnout B_medium@burnout B_fine@burnout` |
| 3 | domain study at M 0.95 and M 1.00 (A_D40@M0.95 and @M1.00 are also production) | `A_D10@M0.95 A_D40@M0.95 A_D10@M1.00 A_D40@M1.00 A_medium@M1.00` |
| 4 | solver cross-check at M 1.20 (the density run is production) | `A_medium@M1.20 A_medium@M1.20:pressure` |
| 4b | B at M 1.20 pressure-based: smooth wall Cp (decision 7) | `B_medium@M1.20:pressure` |
| 5 | validation: sharp cone against Taylor-Maccoll | `cone@burnout cone@M1.20` |
| 6+7 | production, A and B alternating | `A_D40@M0.90 B_D40@M0.90 B_D40@M0.95 A_D40@M0.98 B_D40@M0.98 B_D40@M1.00 A_D40@M1.05 B_D40@M1.05 A_D40@M1.10 B_D40@M1.10 B_medium@M1.20` |
| 8 | production, where A and B cross above M 1 | `A_D40@M1.02 B_D40@M1.02` |
| 8b | domain at M 1.20: is 20 D enough? | `A_D40@M1.20` |

All at once. Time: PLAN.md section 4 estimates the run time per mesh and solver, and 10 h (schedule only) to 26 h
(every run needing its 8 extra blocks) on one VM with 8 processes for the groups planned on 15 Nov; the B mesh
study (2B) and groups 4b, 8 and 8b come on top. Groups can run on several VMs at once (PLAN.md section 4).

```
A_study720@M0.95 A_study720@M1.20:pressure A_study720@M0.95:density A_coarse@M0.95 A_medium@M0.95 A_fine@M0.95 A_coarse@burnout A_medium@burnout A_fine@burnout B_coarse@M0.95 B_medium@M0.95 B_fine@M0.95 B_coarse@burnout B_medium@burnout B_fine@burnout A_D10@M0.95 A_D40@M0.95 A_D10@M1.00 A_D40@M1.00 A_medium@M1.00 A_medium@M1.20 A_medium@M1.20:pressure B_medium@M1.20:pressure cone@burnout cone@M1.20 A_D40@M0.90 B_D40@M0.90 B_D40@M0.95 A_D40@M0.98 B_D40@M0.98 B_D40@M1.00 A_D40@M1.05 B_D40@M1.05 A_D40@M1.10 B_D40@M1.10 B_medium@M1.20 A_D40@M1.02 B_D40@M1.02 A_D40@M1.20
```

Start long queues detached and keep the log, which `residuals.py` reads:

```powershell
Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Python314\python.exe" `
  -ArgumentList "-u run_queue.py A_coarse@M0.95 A_medium@M0.95 A_fine@M0.95" `
  -RedirectStandardOutput runs\queue_1.txt -RedirectStandardError runs\queue_1.err.txt
```

## 5. Numbers and figures, in this order

1. `python check_run.py` → `runs/summary.csv` (convergence and sanity checks of every run).
2. `python residuals.py` → `runs/residuals.csv` (from the queue logs).
3. `python drag_curve.py` → `runs/drag_curve.csv` (nose + coupler) and `python drag_curve.py --field cd_nose` →
   `runs/drag_curve_cd_nose.csv`.
4. Mesh study: `python gci.py A_fine_M0.95 A_medium_M0.95 A_coarse_M0.95` and the same for B and for burnout
   (PLAN.md 2c-2f; burnout uses the mean of the last 500 iterations, as in `fluent_figs_study.py`).
5. `python shock_angle.py cone_burnout cone_M1.20` (cone shock angles); with Ansys's Python
   `taylor_maccoll_targets.py` (the theory) and `mesh_sizes.py` (needs the case files).
6. Figures: with Ansys's Python `report_figures.py` (`figures/fig_cd_mach`); `python fluent_figs.py` (production runs,
   in the Fluent of step 3) and `python fluent_figs.py A_coarse_M0.95 A_fine_M0.95 A_coarse_burnout A_fine_burnout
   A_D10_M0.95 A_D10_M1.00`; `python fluent_figs_study.py` (starts its own hidden Fluent).

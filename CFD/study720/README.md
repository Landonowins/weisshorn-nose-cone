# Fluent study 720: the two 720 mm noses, 2D axisymmetric RANS

Designs of `2027_C_FD_NOSE-CONE-720MM_GEOMETRY.pdf` (both 720 mm long, D = 180 mm, tip radius 4.5 mm):

* **A**: LV-Haack, curved part 679 mm + 41 mm coupler (`geometry_720/nose_lvhaack_L682_c41_rn4.5mm.step`)
* **B**: Von Kármán, curved part 700 mm + 20 mm coupler (`geometry_720/nose_vonkarman_L705_c20_rn4.5mm.step`)

Steady RANS (k-ω SST) in Ansys Fluent 2024 R2 with the study A workflow (`../studyA/README.md`,
REPORT-4), at burnout (M_max) and at M 1.20 and 0.95 of the trajectory, adiabatic walls, plus walls
at 300 K at burnout for the heat flux. Run on 1 Nov 2026 on the DLL VM.

## Results

C_D of `wall_nose` (curved part and blunt tip), A_ref = π D²/4 = 0.025447 m²; pressure + viscous.

| M | A, LV-Haack | B, Von Kármán | B vs A |
|---|---:|---:|---:|
| 0.95 | 0.0320 (0.0032 + 0.0288) | 0.0320 (0.0041 + 0.0278) | ±0 % |
| 1.20 | 0.0949 (0.0686 + 0.0263) | 0.0828 (0.0576 + 0.0252) | −13 % |
| 1.465 (burnout) | 0.0933 (0.0689 + 0.0244) | 0.0791 (0.0556 + 0.0235) | −15 % |

* Against the Python design tool (wave-drag model + flat-plate friction, report 2): +1.7 to +1.9 %
  in total at M 1.2 and burnout for both designs (pressure +2.6 to +2.9 %, friction −1.2 to −0.3 %),
  `python_compare_720.csv`, `figures/fig_python_compare_720`. At M 0.95 the model's transonic ramp
  (onset 0.8, no Euler data below M 1.1) gives 0.055 (A) and 0.047 (B) of wave drag where Fluent
  finds 0.003-0.004; moving the onset to 0.94 reproduces Fluent and changes the apogee by +7 m (A)
  and +2 m (B), so the model was left as it is.
* Heating at burnout, walls at 300 K: stagnation heat flux 172.5 kW/m² (A), 173.4 kW/m² (B), against
  183 kW/m² of the Python estimate; h = q_w / (T_aw − 300 K) is 1888 and 1899 W/(m² K) at the tip and
  550-570 W/(m² K) at the shoulder (`A_h_x.csv`, `B_h_x.csv`). Adiabatic wall temperature 391.3 K
  at the tip (T₀ = 391.6 K), 380-385 K along the nose. Heating does not separate the designs; drag does.
* Coupler friction at burnout: 5.4 N (A, 41 mm), 2.7 N (B, 20 mm); no pressure drag on the cylinder.
* Max y+ (A, burnout) 0.71 on `wall_nose`, 0.28-0.29 on the coupler and body.

## Setup

* Free stream (`fluent_points_python.csv`, from `python_points_720.py`, the trajectory model of
  `studies/total_length.py` with `configs/weisshorn_650.yaml`): burnout M 1.46548, 77 685 Pa,
  273.95 K (t = 9.10 s, 2185 m); M 1.20: 84 801 Pa, 278.55 K; M 0.95: 90 714 Pa, 282.15 K. Design A's
  points are used for both designs, so that A against B compares shapes only (B's own burnout is
  M 1.4706, 0.35 % faster).
* Domain: as study A, 20 D upstream and radially, body pulled to x = 2520 mm (10 D behind the
  coupler), built in Discovery from the 3D STEP. The body line is split at x = 720 mm into
  `wall_coupler` and `wall_body`. `geometry_720/domain2d_*.step` are the same domains generated with
  `../make_domain_2d.py` (area within 0.001 %).
* Mesh (`mesh/A_mesh.msh` 174 380 cells, `mesh/B_mesh.msh` 174 155 cells): study A recipe; triangles,
  50 mm global, 1 mm on the walls (`wall_body` biased ×3, small at x = 720), 0.2 mm sphere of
  influence (r = 15 mm) at the tip, 45 inflation layers from 1 µm, growth 1.2. Max skewness 0.90 / 0.94,
  min orthogonal quality 0.088 / 0.062 (Fluent; worst cells at the 1 → 0.55 mm wall-size step at
  x = 720 and just above the inflation layers).
* Fluent (`setup_fluent.py`): density-based implicit, Roe-FDS, least squares, axisymmetric, k-ω SST,
  ideal gas with three-coefficient Sutherland, operating pressure 0; pressure far field on `inlet` and
  `farfield`, pressure outlet (backflow total T = static T, as study A), I = 1 %, μt/μ = 10.
  500 iterations first order at Courant 1, 500 at Courant 5, then second order at Courant 5.
  M 0.95: pressure-based Coupled with the pseudo-time method, viscous heating on (study A recipe),
  200 iterations first order, then second order. T300 runs restart from the adiabatic solution.
* Convergence: (max − min)/mean of `cd-nose` < 0.1 % over the last 500 iterations (study A rule);
  reached 0.007-0.050 % (`cd_history/`).

## Contents

| Path | Contents |
|---|---|
| `setup_fluent.py`, `run_queue.py` | Fluent setup and the run queue (PyFluent on a running 2024 R2 session) |
| `fluent_figs.py`, `xyfile.py` | Native Fluent figures (REPORT-4 style) from the saved cases |
| `report_figures_720.py` | `fig_cd_mach_720`, `fig_python_compare_720` and `python_compare_720.csv` |
| `python_points_720.py` | Trajectory points (free stream) and Python C_D of both designs |
| `make_h.py` | h(x) from the adiabatic and 300 K wall exports |
| `fluent_cd_720.csv` | Fluent `wall_nose` C_D of every run (wall-forces report) |
| `wall_data/` | Face values on `wall_nose` + `wall_coupler` of each run (Fluent ASCII export) |
| `cd_history/` | C_D (and heat flux for T300) per iteration of each run |
| `figures/` | Contours (Mach, Cp, p, T, tip close-ups), wall plots (Cp, Cf, y+, p, T_wall, q_w, h), C_D history and residuals per run; `*_AB_*` overlays of A and B |
| `mesh/` | The two meshes (Fluent .msh) |

Case and data files (`*.cas.h5`, `*.dat.h5`) are not in git; they are in `CFD/study720/` on the DLL VM.

## Notes and open items

* Cp and the heat flux show a sawtooth between x ≈ 3 and 15 mm, where the 0.2 mm tip cells grow to
  the 1 mm wall cells at the edge of the sphere of influence. Integrated values and the stagnation
  value are not affected; a larger sphere (r ≈ 40 mm) would smooth it.
* Mesh study (coarse / medium / fine) not done for these noses; the medium recipe is study A's.
* Transonic drag rise between M 0.95 and 1.1 not resolved (would need M 1.0 and 1.05).

# Nose-cone design tool (EPFL Rocket Team)

Parametric tool that compares nose-cone profiles for a supersonic sounding rocket and produces the
selected design: wave drag from an axisymmetric Euler CFD database, apogee from a point-mass
trajectory, payload fit, static-margin change, tip and shell heating, a robustness study, a
weighted decision matrix, CAD files (profile CSV, STL, STEP) and publication-quality figures.

It was written for **Weisshorn** (ERT 2027, EuRoC L9, D = 180 mm) and reproduces the design report
`2026_C_FD_REPORT-1_NOSE_CONE` exactly; every input lives in one YAML file, so it can be reused for
another rocket without touching the code. The full documentation is the user guide:
[`docs/user_guide/user_guide.pdf`](docs/user_guide/user_guide.pdf).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m nosecone run configs/weisshorn.yaml          # the design report (about 5 min)
python -m nosecone run configs/example_sweep.yaml      # a smaller parameter sweep
python -m nosecone run configs/example_single_design.yaml
```

Results go to `output/<study>/`: `summary.md` (start here), `tables/`, `figures/`, `cad/`,
`latex/` and `run.log`.

## Reusing it for another rocket

Copy `configs/example_new_rocket.yaml` (or start a file with `base: weisshorn.yaml` and override a
few values), set the vehicle, engine, fins and nose sections, and run it. The CFD database is
non-dimensional (lengths scaled by the base radius), so it is valid for any diameter. Profiles,
fineness ratios or tip radii missing from the database can be computed on the fly
(`aerodynamics.compute_missing: true`, several minutes per geometry).

```bash
python -m nosecone describe-config        # every parameter: unit, valid range, default, meaning
python -m nosecone validate my_rocket.yaml
python -m nosecone profiles               # profile families and built-in profiles
python -m nosecone export my_rocket.yaml  # CAD files of the design only
python -m nosecone cfd my_rocket.yaml     # Euler studies listed in cfd.studies (slow)
```

## Repository layout

| Path | Content |
|---|---|
| `nosecone/` | the package: `geometry`, `euler` (CFD solver), `cfd` (studies, database), `aerodynamics` (wave-drag model), `flight`, `stability`, `heating`, `design` (sweep, robustness, ...), `export` (CSV/STL/STEP), `plotting/`, `reporting`, `config`, `constants`, `pipeline`, `cli` |
| `configs/` | `weisshorn.yaml` (the design report), worked examples, template for another rocket |
| `data/` | CFD database `euler_results.csv`, raw wall-pressure and flow-field files `cfd/`, solver run log |
| `docs/user_guide/` | user guide (LaTeX source and PDF) |
| `report/` | the Weisshorn nose-cone design report (LaTeX source and PDF) |
| `tests/` | `verify_reference.py` checks the tool against the design report; reference outputs |
| `legacy/scripts/` | the original one-off scripts the tool was refactored from (kept for traceability) |

## Verification

```bash
python tests/verify_reference.py             # tables, numbers, CAD (about 2 min)
python tests/verify_reference.py --figures   # + every figure, pixel by pixel (needs LaTeX)
python tests/verify_reference.py --cfd       # + three CFD cases recomputed bit for bit
```

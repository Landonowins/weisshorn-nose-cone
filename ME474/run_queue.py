"""Run the ME-474 cases one after the other on the Fluent session of the newest server_info*.txt here.

Usage: python run_queue.py SPEC [SPEC ...]
  SPEC = MESH@POINT[:SOLVER]
    MESH    a mesh in meshes/ without .msh (A_coarse, A_medium, A_fine, B_medium, A_D10, A_D40, cone ...),
            or A_study720 / B_study720: the Rocket Team meshes of CFD/study720/mesh (174k cells)
    POINT   a row of fluent_points.csv: M0.95 M0.98 M1.00 M1.02 M1.05 M1.10 M1.20 burnout
    SOLVER  pressure (pressure-based Coupled) or density (density-based Roe-FDS). Default: pressure
            up to M 1.10, density from M 1.20. Give it only to override, e.g. A_medium@M1.20:pressure
  The run is called MESH_POINT, plus _SOLVER when the solver is not the default. Its files go to
  runs/<run>/: run.json, <run>_cd.out, <run>_flux.out, case/data, <run>_wall.csv, <run>_axis.csv,
  <run>_report.txt.
  finish:RUN   continue the case already in the session until converged, then save and export.

Each case: read the mesh, apply setup_fluent.py at the free stream of POINT, hybrid initialisation,
the solver's schedule (pressure: 200 first + 1000 second order; density: 500 + 500 first order,
then 3000 second order), then 500-iteration blocks until cd-nose changes by less than 0.1 % over
the last 500 iterations (at most 8 blocks), then save and export. check_run.py summarises the runs.
"""
import csv
import json
import sys

import setup_fluent as F
from check_run import mesh_path, read_report

HERE = F.HERE
POINTS = {r["point"]: r for r in csv.DictReader(open(HERE / "fluent_points.csv"))}


def parse(spec):
    mesh, _, rest = spec.partition("@")
    point, _, solver = rest.partition(":")
    if point not in POINTS:
        raise SystemExit(f"{spec}: unknown point {point!r}, choose from {list(POINTS)}")
    if not mesh_path(mesh).exists():
        raise SystemExit(f"{spec}: no mesh {mesh_path(mesh)}")
    default = "pressure" if float(POINTS[point]["mach"]) <= 1.10 + 1e-9 else "density"
    solver = solver or default
    if solver not in ("pressure", "density"):
        raise SystemExit(f"{spec}: solver must be pressure or density")
    run = f"{mesh}_{point}" + ("" if solver == default else f"_{solver}")
    return run, mesh, point, solver


def cd_nose_change(run, window=500):
    rows = read_report(F.run_dir(run) / f"{run}_cd.out")
    its = sorted(rows)[-window:]
    vals = [rows[i]["cd-nose"] for i in its]
    return its[-1], vals[-1], 100 * (max(vals) - min(vals)) / (sum(vals) / len(vals))


def converge(s, run, block=500, max_blocks=8):
    it, cd, ch = cd_nose_change(run)
    for _ in range(max_blocks):
        if ch < 0.1:
            break
        s.settings.solution.run_calculation.iterate(iter_count=block)
        it, cd, ch = cd_nose_change(run)
        print(f"{run}: iteration {it}, cd-nose {cd:.5f}, change over last 500 {ch:.3f} %", flush=True)
    return it, cd, ch


def finish(s, run):
    """Converge, save and export the case already in the session."""
    it, cd, ch = converge(s, run)
    s.settings.file.write(file_type="case-data", file_name=(F.run_dir(run) / run).as_posix())
    F.export_wall(s, run)
    F.export_axis(s, run)
    F.reports(s, run)
    flag = "" if ch < 0.1 else "  NOT CONVERGED (cd-nose criterion)"
    print(f"DONE {run}: iteration {it}, cd-nose {cd:.5f}, change over last 500 {ch:.3f} %{flag}", flush=True)


def run_case(s, spec):
    run, mesh, point, solver = parse(spec)
    fs = POINTS[point]
    p, m, t = float(fs["p_Pa"]), float(fs["mach"]), float(fs["T_K"])
    meta = dict(run=run, mesh=mesh, point=point, solver=solver, p_inf=p, mach=m, t_inf=t,
                t0=float(fs["T0_K"]), p0=float(fs["p0_Pa"]), p0_pitot=float(fs["p0_pitot_Pa"]))
    (F.run_dir(run) / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"\n=== {run}: mesh {mesh}, {point} (M {m}, p {p} Pa, T {t} K), {solver}-based", flush=True)
    s.settings.file.read_case(file_name=mesh_path(mesh).as_posix())
    F.general(s)
    if solver == "pressure":
        F.pressure_based(s)
    F.bcs(s, run=run, p_inf=p, m_inf=m, t_inf=t)
    if solver == "pressure":
        F.pb_order(s, "first")
        F.init(s, run)
        F.schedule_pb(s, run)
    else:
        F.methods(s)
        F.init(s, run)
        F.schedule(s, run)
    finish(s, run)


if __name__ == "__main__":
    specs = sys.argv[1:]
    for spec in specs:  # check every spec before starting Fluent work
        if not spec.startswith("finish:"):
            parse(spec)
    s = F.connect()
    for spec in specs:
        if spec.startswith("finish:"):
            finish(s, spec.split(":", 1)[1])
        else:
            run_case(s, spec)

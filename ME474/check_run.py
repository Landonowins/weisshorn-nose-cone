"""Convergence and sanity checks of the runs in runs/, one row per run, to runs/summary.csv.

Needs no Fluent: it reads what run_queue.py leaves in runs/<run>/ (run.json, <run>_cd.out,
<run>_flux.out, <run>_wall.csv, <run>_axis.csv, <run>_report.txt) and the mesh headers.
Run it from anywhere: paths are relative to this file.

Columns (missing inputs give empty cells):
  cells            cell count, from the header of meshes/<mesh>.msh
  iterations       last iteration in <run>_cd.out
  cd_nose, cd_total  C_D of wall_nose and of all walls, last iteration
  cd_spread_pct    (max - min) / mean of cd-nose over the last 500 iterations   [target < 0.1 %]
  cd_p, cd_v       pressure and viscous C_D of wall_nose, from the wall-forces transcript
  mass_imb_pct     |net mass flux| / |inlet mass flux|, last iteration           [target < 0.2 %]
  heat_imb_pct     |net heat flux| / |inlet heat flux|, last iteration           [target < 0.2 %]
  p0_inlet_pct     total pressure at the upstream end of the axis vs the free-stream p0
  p0_axis_max_pct  highest total pressure along the axis vs the free-stream p0 (must not be > 0)
  p_stag_pct       highest wall pressure vs the theoretical stagnation pressure (isentropic
                   below M 1, behind a normal shock above)                     [physical: <= 0]
  yplus_max        on wall_nose + wall_coupler                                  [target < 1]
  min_orth_q, max_aspect  from the mesh quality transcript

Usage: python check_run.py [RUN ...]    (default: every run in runs/)
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
STUDY720 = HERE.parent / "CFD" / "study720" / "mesh"   # the Rocket Team meshes, used as they are
COLS = ["run", "mesh", "cells", "point", "mach", "solver", "iterations", "cd_nose", "cd_total", "cd_spread_pct",
        "cd_p", "cd_v", "mass_imb_pct", "heat_imb_pct", "p0_inlet_pct", "p0_axis_max_pct", "p_stag_pct",
        "yplus_max", "min_orth_q", "max_aspect"]
NUM = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"


def read_report(path, default=("cd-total", "cd-nose")):
    """Iteration -> {report name: value} from a Fluent report file (.out). Column names from its
    '"Iteration" "..."' header line, else `default` (the layout of the Rocket Team cd files)."""
    names, rows = None, {}
    for line in open(path):
        if names is None and '"Iteration"' in line:
            names = re.findall(r'"([^"]+)"', line)[1:]
            continue
        parts = line.split()
        if parts and parts[0].isdigit():
            cols = names or list(default)
            if len(parts) > len(cols):
                rows[int(parts[0])] = {n: float(v) for n, v in zip(cols, parts[1:])}
    return rows


def read_export(path):
    """Fluent ASCII export -> list of {column: float}."""
    with open(path) as fh:
        rd = csv.reader(fh)
        head = [h.strip() for h in next(rd)]
        return [{h: float(v) for h, v in zip(head, r)} for r in rd if len(r) == len(head)]


def mesh_path(mesh):
    """meshes/<mesh>.msh, except A_study720 / B_study720: the meshes of CFD/study720/mesh."""
    if mesh in ("A_study720", "B_study720"):
        return STUDY720 / f"{mesh[0]}_mesh.msh"
    return HERE / "meshes" / f"{mesh}.msh"


def mesh_cells(mesh):
    """Cell count from the '(12 (0 1 <last cell, hex> 0))' line of a Fluent .msh header."""
    path = mesh_path(mesh)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        m = re.search(rb"\(12 \(0 1 ([0-9a-fA-F]+) 0\)\)", fh.read(20000))
    return int(m.group(1), 16) if m else None


def check(d):
    run = d.name
    meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {"run": run}
    row = {k: meta.get(k, "") for k in ("run", "mesh", "point", "mach", "solver")}
    row["cells"] = mesh_cells(meta["mesh"]) if "mesh" in meta else None

    f = d / f"{run}_cd.out"
    if f.exists():
        rows = read_report(f)
        its = sorted(rows)
        if its:
            last = rows[its[-1]]
            vals = [rows[i]["cd-nose"] for i in its[-500:]]
            row.update(iterations=its[-1], cd_nose=f"{last['cd-nose']:.5f}", cd_total=f"{last['cd-total']:.5f}",
                       cd_spread_pct=f"{100 * (max(vals) - min(vals)) / (sum(vals) / len(vals)):.3f}")

    f = d / f"{run}_flux.out"
    if f.exists():
        rows = read_report(f, default=())
        if rows:
            last = rows[max(rows)]
            for tag, col in (("m", "mass_imb_pct"), ("q", "heat_imb_pct")):
                if f"{tag}-net" in last and last.get(f"{tag}-inlet"):
                    row[col] = f"{100 * abs(last[f'{tag}-net'] / last[f'{tag}-inlet']):.4f}"

    f = d / f"{run}_axis.csv"
    if f.exists() and "p0" in meta:
        ax = sorted(read_export(f), key=lambda r: r["x-coordinate"])
        if ax:
            row["p0_inlet_pct"] = f"{100 * (ax[0]['total-pressure'] / meta['p0'] - 1):+.3f}"
            row["p0_axis_max_pct"] = f"{100 * (max(r['total-pressure'] for r in ax) / meta['p0'] - 1):+.3f}"

    f = d / f"{run}_wall.csv"
    if f.exists():
        w = read_export(f)
        if "p0_pitot" in meta:
            row["p_stag_pct"] = f"{100 * (max(r['pressure'] for r in w) / meta['p0_pitot'] - 1):+.3f}"
        row["yplus_max"] = f"{max(r['y-plus'] for r in w):.3f}"

    f = d / f"{run}_report.txt"
    if f.exists():
        txt = f.read_text(errors="ignore")
        for line in txt.splitlines():
            parts = line.split()
            if parts and parts[0] == "wall_nose":
                nums = [p for p in parts[1:] if re.fullmatch(NUM, p)]
                if len(nums) >= 6:  # pressure, viscous, total force, then the same as coefficients
                    row["cd_p"], row["cd_v"] = nums[-3], nums[-2]
        m = re.search(rf"Minimum Orthogonal Quality\s*=\s*({NUM})", txt)
        row["min_orth_q"] = m.group(1) if m else ""
        m = re.search(rf"Maximum Aspect Ratio\s*=\s*({NUM})", txt)
        row["max_aspect"] = m.group(1) if m else ""
    return row


def main():
    dirs = [RUNS / r for r in sys.argv[1:]] or sorted(p for p in RUNS.glob("*") if p.is_dir())
    rows = [check(d) for d in dirs]
    if not sys.argv[1:]:
        with open(RUNS / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, COLS)
            wr.writeheader()
            wr.writerows(rows)
    for r in rows:
        print(", ".join(f"{k} {r.get(k)}" for k in COLS if r.get(k) not in (None, "")))


if __name__ == "__main__":
    main()

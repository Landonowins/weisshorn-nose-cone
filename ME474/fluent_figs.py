"""Native Fluent figures of the ME-474 runs, in the style of CFD/study720/fluent_figs.py: 2400 x 1500 PNG, white background.

Usage: python fluent_figs.py [RUN ...]     default: the production set (drag_curve.py) and both M 1.20 pressure-based runs
       python fluent_figs.py --overlays    only the wall overlays (on whatever case is loaded)

Per run (runs/<run>/<run>.cas.h5 + .dat.h5): Mach field around the nose and wide (bow shock and sonic region), Cp field,
Mach close-up of the tip, the residual history and the cd-nose history.
Overlays, after the runs: Cp, Cf and y+ along wall_nose + wall_coupler, A against B at every production point;
A and B at M 1.20 density- against pressure-based (the Roe zigzag, PLAN.md section 2h); A against B pressure-based at
M 1.20. The wall data come from each run's <run>_wall.csv (face values, sorted by x), so the overlays need no case
of their own.
Writes figures/fluent_*.png and the XY files in figures/xy/.

Runs on the Fluent session of the newest server_info*.txt here, so it refuses to start while run_queue.py is running.
"""
import csv
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
XY = FIG / "xy"
sys.path.insert(0, str(HERE))
import drag_curve as dc  # noqa: E402
import setup_fluent as F  # noqa: E402

# short name, column of <run>_wall.csv = Fluent field (gives the title and unit), axis label, decimals
WALL = [("cp", "pressure-coefficient", "Pressure coefficient Cp", 2),
        ("cf", "skin-friction-coef", "Skin friction coefficient Cf", 4),
        ("yplus", "y-plus", "Wall y+", 2)]
CAMERA = {"full": (0.30, 0.0, 1.2, 0.75), "wide": (0.0, 0.0, 4.0, 2.5), "tip": (0.004, 0.0, 0.024, 0.015)}
CP_RANGE = (-0.3, 0.6)  # study720
SHAPE_LABEL = {"A": "A, LV-Haack", "B": "B, Von Karman"}
EXTRA = ["A_medium_M1.20_pressure", "B_medium_M1.20_pressure"]  # for the M 1.20 Cp(x) without the Roe zigzag


def mach_range(mach):
    """One range for the drag-rise points, M = 1 at mid-scale so that the sonic line shows (the bow shock stands
    146 / 50 / 18 mm ahead of the tip at M 1.02 / 1.05 / 1.10); the study A ranges at M 1.20 and burnout."""
    return (0.7, 1.3) if mach <= 1.10 else (0.8, 1.4) if mach < 1.3 else (1.0, 1.6)


def queue_running():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*run_queue.py*' } | Select-Object -ExpandProperty ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return out.split()


def save(s, name):
    pic = s.settings.results.graphics.picture
    pic.use_window_resolution = False
    pic.x_resolution = 2400
    pic.y_resolution = 1500
    pic.invert_background = True
    pic.save_picture(file_name=str(FIG / name))
    print("saved", name, flush=True)


def camera(s, cx, cy, w, h):
    for cmd in ["/views/camera/projection orthographic", f"/views/camera/target {cx} {cy} 0",
                f"/views/camera/position {cx} {cy} 10", "/views/camera/up-vector 0 1 0",
                f"/views/camera/field {w} {h}"]:
        s.execute_tui(cmd)


def contour(s, field, fmt, rng, view, name):
    c = s.settings.results.graphics.contour
    if name not in c.get_object_names():
        c.create(name=name)
    o = c[name]
    o.field = field
    o.surfaces_list = []
    o.coloring = {"option": "smooth"}
    o.color_map.format = fmt
    o.range_option.option = "auto-range-off"
    o.range_option.auto_range_off.minimum = rng[0]
    o.range_option.auto_range_off.maximum = rng[1]
    s.settings.results.graphics.views.mirror_zones = ["axis"]
    o.display()
    camera(s, *CAMERA[view])
    save(s, f"fluent_{name}.png")  # the object name is the label under the colour bar


def xy_plot(s, name):
    """A fresh XY plot: one that exists already would keep the files read into it before."""
    plots = s.settings.results.plot.xy_plot
    if name in plots.get_object_names():
        del plots[name]
    plots.create(name=name)
    return plots[name]


def file_plot(s, name, func, curves, ylabel, prec, out):
    """XY plot of Fluent XY files [(path, colour, pattern)]; the y function sets the title and unit, the data come from
    the files. Fluent makes the last file read curve 0 (first in the legend), so the files are read in reverse."""
    p = xy_plot(s, name)
    p.y_axis_function = func
    p.surfaces_list = []
    for path, _, _ in reversed(curves):
        p.read_from_file(filename=str(path))
    p.axes.labels = {"x_axis": "Position x", "y_axis": ylabel}
    p.axes.numbers = {"x_format": "float", "x_axis_precision": 2, "y_format": "float", "y_axis_precision": prec}
    p.axes.rules = {"x_axis": {"draw_major_rules": True}, "y_axis": {"draw_major_rules": True}}
    for i, (_, color, pattern) in enumerate(curves):
        p.curves.lines[i] = {"pattern": pattern, "weight": 3, "color": color}
        p.curves.markers[i] = {"symbol": "", "size": 0.3, "color": color}
    p.display()
    save(s, out)


def write_xy(path, label, ylabel, pts):
    with open(path, "w") as f:
        f.write(f'(title "{label}")\n(labels "Position x" "{ylabel}")\n\n((xy/key/label "{label}")\n')
        f.write("".join(f"{a:.6g}\t{b:.6g}\n" for a, b in pts))
        f.write(")\n")


def wall_xy(run, col, label, tag):
    """One column of runs/<run>/<run>_wall.csv (wall_nose + wall_coupler faces) against x, sorted, as an XY file."""
    with open(dc.RUNS / run / f"{run}_wall.csv") as fh:
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    path = XY / f"{col}_{tag}_{run}.xy"
    write_xy(path, label, col, sorted((float(r["x-coordinate"]), float(r[col])) for r in rows))
    return path


def cd_plot(s, run, cd_final):
    """The cd-nose report plot, y range 0 to at least twice the converged value, in steps of 0.08 (Fluent draws 8
    intervals, so the ticks come out at multiples of 0.01)."""
    rps = s.settings.solution.monitor.report_plots
    rps["cd-nose-rplot"] = {"report_defs": ["cd-nose"]}
    rp = rps["cd-nose-rplot"]
    rp.axes.y = {"auto_range": False, "min": 0.0, "max": 0.08 * math.ceil(2 * cd_final / 0.08), "show_major_gridlines": True,
                 "label": "Cd (wall_nose)", "number_format": {"format_type": "float", "precision": 3}}
    rp.axes.x = {"label": "Iteration", "show_major_gridlines": True}
    rp.curves[0].line = {"pattern": "----", "weight": 3, "color": "blue"}
    rp.plot()
    save(s, f"fluent_cd_{run}.png")


def display_prefs(s):
    for cmd in ("/display/set/windows/logo no", "/preferences/graphics/boundary-markers/enabled no"):
        try:
            s.execute_tui(cmd)
        except Exception:
            print("not accepted:", cmd, flush=True)


def figures(s, run, summary):
    meta = json.loads((dc.RUNS / run / "run.json").read_text())
    s.settings.file.read_case_data(file_name=(dc.RUNS / run / f"{run}.cas.h5").as_posix())
    display_prefs(s)
    mr = mach_range(meta["mach"])
    jobs = [("mach", lambda: contour(s, "mach-number", "%0.2f", mr, "full", f"mach_{run}")),
            ("mach wide", lambda: contour(s, "mach-number", "%0.2f", mr, "wide", f"mach_wide_{run}")),
            ("cpfield", lambda: contour(s, "pressure-coefficient", "%0.2f", CP_RANGE, "full", f"cpfield_{run}")),
            ("mach tip", lambda: contour(s, "mach-number", "%0.2f", (0.0, mr[1]), "tip", f"mach_tip_{run}")),
            ("residuals", lambda: (s.settings.solution.monitor.residual.plot(), save(s, f"fluent_residuals_{run}.png"))),
            ("cd", lambda: cd_plot(s, run, float(summary[run]["cd_nose"])))]
    for what, job in jobs:
        try:
            job()
        except Exception:
            print(f"FAILED {what} {run}\n" + traceback.format_exc(), flush=True)


def overlays(s, summary):
    groups = []
    for point in dc.POINTS:
        runs = [dc.run_name(sh, point) for sh in "AB"]
        if all(r in summary for r in runs):
            groups.append((f"AB_{point}", [(runs[0], SHAPE_LABEL["A"]), (runs[1], SHAPE_LABEL["B"])]))
    for sh in "AB":  # M 1.20, density- vs pressure-based (the zigzag)
        dens, pres = f"{sh}_medium_M1.20", f"{sh}_medium_M1.20_pressure"
        if dens in summary and pres in summary:
            groups.append((f"solver_{sh}_M1.20", [(dens, "density-based (Roe)"), (pres, "pressure-based (Coupled)")]))
    if all(f"{sh}_medium_M1.20_pressure" in summary for sh in "AB"):  # smooth A-vs-B wall Cp at M 1.20
        groups.append(("AB_M1.20_pressure", [("A_medium_M1.20_pressure", SHAPE_LABEL["A"]),
                                             ("B_medium_M1.20_pressure", SHAPE_LABEL["B"])]))
    for tag, members in groups:
        # A and B as in fig_cd_mach: LV-Haack green dash-dot, Von Karman indigo solid (Fluent has no purple)
        styles = ([("red", "----"), ("dark blue", "----")] if tag.startswith("solver")
                  else [("dark green", "-.-."), ("dark blue", "----")])
        for var, col, ylabel, prec in WALL:
            curves = [(wall_xy(run, col, label, tag), c, pat) for (run, label), (c, pat) in zip(members, styles)]
            try:
                file_plot(s, f"o-{var}-{tag}", col, curves, ylabel, prec, f"fluent_{var}_{tag}.png")
            except Exception:
                print(f"FAILED overlay {var} {tag}\n" + traceback.format_exc(), flush=True)


def main(args):
    busy = queue_running()
    if busy:
        sys.exit(f"run_queue.py is running (pid {' '.join(busy)}): wait for it, the figures read cases into its Fluent.")
    FIG.mkdir(exist_ok=True)
    XY.mkdir(exist_ok=True)
    summary = {r["run"]: r for r in csv.DictReader(open(dc.RUNS / "summary.csv"))}
    s = F.connect()
    if args == ["--overlays"]:
        overlays(s, summary)
        return
    runs = args or [dc.run_name(sh, p) for p in dc.POINTS for sh in "AB"] + EXTRA
    runs = [r for r in runs if (dc.RUNS / r / f"{r}.dat.h5").exists() and r in summary]
    for run in runs:
        print(f"=== {run}", flush=True)
        figures(s, run, summary)
    if not args:
        overlays(s, summary)


if __name__ == "__main__":
    main(sys.argv[1:])

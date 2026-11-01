"""Native Fluent figures of the 720 mm runs, in the style of the study A figures of REPORT-4
(CFD/runs/figwork/fluent_plot_scripts/fluent_plots.py): 2400 x 1500 PNG, white background.

Per run: Mach, Cp, static pressure and temperature contours (study A camera and ranges), tip
close-ups of Mach and temperature, Cp / Cf / y+ / static pressure / wall temperature along
wall_nose, the cd-nose history and the residuals. Runs with walls at 300 K also get the heat flux
into the wall and the heat transfer coefficient h = q_w / (T_aw - 300 K) along the nose (from
RUN_h_x.csv, written by make_h.py). The overlay mode puts several runs on one wall plot.
Runs on a second, post-processing Fluent (server_info-post.txt) so the solver queue keeps going.

Usage: python fluent_figs.py RUN [RUN ...]            per-run figures, e.g. A_adiab B_adiab A_T300
       python fluent_figs.py --overlay TAG RUN RUN ...  wall plots of RUNs together, fluent_<var>_<TAG>.png
"""
import sys
import traceback
from pathlib import Path

import ansys.fluent.core as pf

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
XY = FIG / "xy"
SI = HERE / "server_info-post.txt"
sys.path.insert(0, str(HERE))
from xyfile import sort_xy  # noqa: E402

# short name, Fluent field, axis label, decimals
# short name, Fluent field, axis label (Fluent appends the field's unit), decimals
WALL = [("cp", "pressure-coefficient", "Pressure coefficient Cp", 2),
        ("cf", "skin-friction-coef", "Skin friction coefficient Cf", 4),
        ("yplus", "y-plus", "Wall y+", 2),
        ("p", "pressure", "Static pressure", 0),
        ("twall", "temperature", "Wall temperature", 1)]
# short name, Fluent field that gives the title and unit, axis label, decimals, column of RUN_h_x.csv
HEAT = [("qw", "heat-flux", "Heat flux into the wall q_w", 0, 3),
        ("h", "heat-transfer-coef", "Heat transfer coefficient h", 0, 4)]
CD_RANGE = {"M0.95": (0.0, 0.10), "M1.20": (0.05, 0.25), "burnout": (0.05, 0.25)}
MACH_RANGE = {"M0.95": (0.6, 1.2), "M1.20": (0.8, 1.4), "burnout": (1.0, 1.6)}  # study A ranges
CAMERA = {"full": (0.30, 0.0, 1.2, 0.75), "tip": (0.004, 0.0, 0.024, 0.015)}
COLORS = ["blue", "red", "dark green", "magenta"]


def connect():
    address, password = SI.read_text().split()
    ip, _, port = address.rpartition(":")
    try:
        return pf.connect_to_fluent(ip="127.0.0.1", port=int(port), password=password, cleanup_on_exit=False)
    except Exception:
        return pf.connect_to_fluent(ip=ip, port=int(port), password=password, allow_remote_host=True,
                                    insecure_mode=True, cleanup_on_exit=False)


s = connect()
r = s.settings.results


def save(name):
    pic = r.graphics.picture
    pic.use_window_resolution = False
    pic.x_resolution = 2400
    pic.y_resolution = 1500
    pic.invert_background = True
    pic.save_picture(file_name=str(FIG / name))
    print("saved", name, flush=True)


def xy(name):
    plots = r.plot.xy_plot
    if name not in plots.get_object_names():
        plots.create(name=name)
    return plots[name]


def file_plot(name, func, paths, xlabel, ylabel, prec, out):
    """XY plot of Fluent XY files; the y function sets the title and unit, the data come from the files."""
    p = xy(name)
    p.y_axis_function = func
    p.surfaces_list = []
    for path in paths:
        p.read_from_file(filename=str(path))
    p.axes.labels = {"x_axis": xlabel, "y_axis": ylabel}
    p.axes.numbers = {"x_format": "float", "x_axis_precision": 2, "y_format": "float", "y_axis_precision": prec}
    p.axes.rules = {"x_axis": {"draw_major_rules": True}, "y_axis": {"draw_major_rules": True}}
    for i in range(len(paths)):
        p.curves.lines[i] = {"pattern": "----", "weight": 3, "color": COLORS[i]}
        p.curves.markers[i] = {"symbol": "", "size": 0.3, "color": COLORS[i]}
    p.display()
    save(out)


def write_xy(path, label, xlabel, ylabel, pts):
    with open(path, "w") as f:
        f.write(f'(title "{label}")\n(labels "{xlabel}" "{ylabel}")\n\n((xy/key/label "{label}")\n')
        f.write("".join(f"{a:.6g}\t{b:.6g}\n" for a, b in pts))
        f.write(")\n")


def camera(cx, cy, w, h):
    for cmd in ["/views/camera/projection orthographic", f"/views/camera/target {cx} {cy} 0",
                f"/views/camera/position {cx} {cy} 10", "/views/camera/up-vector 0 1 0",
                f"/views/camera/field {w} {h}"]:
        s.execute_tui(cmd)


def contour(field, fmt, rng, view, name):
    c = r.graphics.contour
    if name not in c.get_object_names():
        c.create(name=name)
    o = c[name]
    o.field = field
    o.surfaces_list = []
    o.coloring = {"option": "smooth"}
    o.color_map.format = fmt
    if rng:
        o.range_option.option = "auto-range-off"
        o.range_option.auto_range_off.minimum = rng[0]
        o.range_option.auto_range_off.maximum = rng[1]
    else:
        o.range_option.option = "auto-range-on"
    r.graphics.views.mirror_zones = ["axis"]
    o.display()
    camera(*CAMERA[view])
    save(f"fluent_{name}.png")  # the object name is the label under the colour bar, as in study A


def wall_xy(run, var, func):
    w = xy("w-" + var)
    w.options.node_values = False
    w.plot_direction.option = "direction-vector"
    w.plot_direction.direction_vector = {"x_component": 1, "y_component": 0}
    w.y_axis_function = func
    w.surfaces_list = ["wall_nose"]
    raw, srt = XY / f"{var}_{run}_raw.xy", XY / f"{var}_{run}.xy"
    w.write_to_file(filename=str(raw))
    sort_xy(str(raw), str(srt), run)
    return srt


def cd_plot(run, point):
    """cd-nose history from the report plot, as the study A figures (y range fixed, as CD_RANGE there)."""
    rps = s.settings.solution.monitor.report_plots
    rps["cd-nose-rplot"] = {"report_defs": ["cd-nose"]}
    rp = rps["cd-nose-rplot"]
    lo, hi = CD_RANGE[point]
    rp.axes.y = {"auto_range": False, "min": lo, "max": hi, "show_major_gridlines": True,
                 "label": "Cd (wall_nose)", "number_format": {"format_type": "float", "precision": 3}}
    rp.axes.x = {"label": "Iteration", "show_major_gridlines": True}
    rp.curves[0].line = {"pattern": "----", "weight": 3, "color": "blue"}
    rp.plot()
    save(f"fluent_cd_{run}.png")


def cd_history(run):
    rows = {}
    for line in open(HERE / f"{run}_cd.out"):
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows[int(parts[0])] = float(parts[2])
    path = XY / f"cd_{run}.xy"
    write_xy(path, run, "Iteration", "Cd (wall_nose)", [(i, rows[i]) for i in sorted(rows)])
    return path


def heat_xy(run, var, col):
    """q_w or h of RUN_h_x.csv (x in mm) as a Fluent XY file in metres."""
    rows = [line.split(",") for line in open(HERE / f"{run[0]}_h_x.csv")][1:]
    path = XY / f"{var}_{run}.xy"
    write_xy(path, run, "Position x", var, sorted((float(r[0]) / 1e3, float(r[col])) for r in rows if r[0].strip()))
    return path


def figures(run, point):
    s.settings.file.read_case_data(file_name=(HERE / f"{run}.cas.h5").as_posix())
    jobs = []
    for var, func, ylabel, prec in WALL:
        jobs.append((var, lambda var=var, func=func, ylabel=ylabel, prec=prec: file_plot(
            "f-" + var, func, [wall_xy(run, var, func)], "Position x", ylabel, prec, f"fluent_{var}_{run}.png")))
    if "T300" in run:
        for var, func, ylabel, prec, col in HEAT:
            jobs.append((var, lambda var=var, func=func, ylabel=ylabel, prec=prec, col=col: file_plot(
                "f-" + var, func, [heat_xy(run, var, col)], "Position x", ylabel, prec, f"fluent_{var}_{run}.png")))
    jobs.append(("cd", lambda: cd_plot(run, point)))
    jobs.append(("residuals", lambda: (s.settings.solution.monitor.residual.plot(),
                                       save(f"fluent_residuals_{run}.png"))))
    t_rng = (270.0, 395.0)
    jobs += [
        ("mach", lambda: contour("mach-number", "%0.2f", MACH_RANGE[point], "full", f"mach_{run}")),
        ("cpfield", lambda: contour("pressure-coefficient", "%0.2f", (-0.3, 0.6), "full", f"cpfield_{run}")),
        ("pfield", lambda: contour("pressure", "%0.0f", None, "full", f"pfield_{run}")),
        ("tempfield", lambda: contour("temperature", "%0.0f", t_rng, "full", f"tempfield_{run}")),
        ("mach tip", lambda: contour("mach-number", "%0.2f", (0.0, MACH_RANGE[point][1]), "tip", f"mach_tip_{run}")),
        ("temp tip", lambda: contour("temperature", "%0.0f", t_rng, "tip", f"temp_tip_{run}")),
    ]
    for what, job in jobs:
        try:
            job()
        except Exception:
            print(f"FAILED {what} {run}\n" + traceback.format_exc(), flush=True)


def overlay(tag, runs):
    heat = all("T300" in run for run in runs)
    for var, func, ylabel, prec in WALL:
        try:
            file_plot("o-" + var, func, [XY / f"{var}_{run}.xy" for run in runs], "Position x", ylabel, prec,
                      f"fluent_{var}_{tag}.png")
        except Exception:
            print(f"FAILED overlay {var}\n" + traceback.format_exc(), flush=True)
    for var, func, ylabel, prec, col in HEAT if heat else []:
        try:
            file_plot("o-" + var, func, [heat_xy(run, var, col) for run in runs], "Position x", ylabel, prec,
                      f"fluent_{var}_{tag}.png")
        except Exception:
            print(f"FAILED overlay {var}\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    XY.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "--overlay":
        overlay(sys.argv[2], sys.argv[3:])
    else:
        for run in sys.argv[1:]:
            point = "M0.95" if "M0.95" in run else "M1.20" if "M1.20" in run else "burnout"
            figures(run, point)

"""Native Fluent figures for the report sections before Results: geometry, mesh, mesh and domain study, cone validation.

Usage: python fluent_figs_study.py [geometry] [mesh] [studies] [history] [cone]     (default: all five)

Starts its own Fluent 2024 R2 (hidden GUI, 4 processes) and closes it at the end, so it never touches an open Fluent
session or a running queue. Writes figures/fluent_*.png (2400 x 1500, white background, as fluent_figs.py) and the XY
files in figures/xy/. About 5 minutes.

geometry  fluent_geometry_profiles   r(x) of both noses and couplers (CFD/geometry_720), shoulders and x = 720 mm
          fluent_geometry_tip        the first 30 mm: 4.5 mm tip sphere, where the profiles part
          fluent_geometry_domains    the 10 D, 20 D and 40 D domains and their boundary types (PLAN.md section 6)
mesh      fluent_mesh_<view>_<mesh>  Fluent's own mesh display, no mirror, with Fluent's scale bar (ruler) at the
                                     bottom centre; width of the view (= picture width): domain 12.8 m (40 D) or
                                     6.4 m (20 D); nose 0.88 m; tip 112 mm (the 40 mm tip sphere); tipzoom 16 mm
                                     (inflation layers on the 4.5 mm tip); wall 48 mm at x = 0.345 m (45 layers,
                                     18.3 mm); shoulder 80 mm around x = 0.70 m; cone tipzoom 0.4 mm (the sliver fan
                                     at the sharp tip). A and B medium, A coarse and fine (nose, tip, wall) for the
                                     mesh study. The ruler takes the foreground colour: black here (Fluent's default
                                     draws it light grey on the white background).
studies   fluent_meshstudy_M0.95     C_D of the nose against h/h_fine (A and B), A's Richardson value and GCI band
          fluent_meshstudy_change    the four mesh studies as change from the fine mesh in %
          fluent_domainstudy_M0.95, fluent_domainstudy_M1.00, fluent_domainstudy_change   C_D against domain size L/D
history   fluent_cd_history_burnout_{A,B}  C_D of the nose over the last 1000 iterations of the burnout mesh study:
          the coarse meshes oscillate (0.10 % A, 0.13 % B peak to peak), so they ran to 8000 iterations and the mesh
          study takes the mean of the last 500; medium and fine are flat (fluent_cd_<run> spans 0 to 2 C_D, too wide
          to show it)
cone      fluent_cp_cone_{burnout,M1.20}   wall Cp of the sharp cone against Taylor-Maccoll (PLAN.md section 2h).
          The cone's Mach and Cp fields come from `python fluent_figs.py cone_burnout cone_M1.20` (in the GUI session:
          in this hidden session Fluent draws the colour-bar text far too large).

Values: mesh study at M 0.95 = last iteration (runs/summary.csv, as PLAN.md 2c and 2e); at burnout = mean of the last
500 iterations (PLAN.md 2d, 2f: the coarse meshes oscillate); Richardson extrapolation and GCI from gci.py. Domain study
= runs/summary.csv. Fluent titles an XY plot "<y function> vs. <x function>", so the axis functions here are custom field
functions whose names say what is plotted; their values are never used (the data come from the XY files). Fluent
cannot hide that title, and a function name with spaces breaks Fluent (the define command fails and Fluent hangs), so
the names are hyphenated like Fluent's own ("pressure-coefficient").
"""
import csv
import math
import sys
import tempfile
from pathlib import Path

import ansys.fluent.core as pf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_run as cr  # noqa: E402
import drag_curve as dc  # noqa: E402
import fluent_figs as ff  # noqa: E402
import gci  # noqa: E402

GEOM = HERE.parent / "CFD" / "geometry_720"
PROFILE = {"A": ("nose_lvhaack_L682_c41_rn4.5mm.csv", 0.679), "B": ("nose_vonkarman_L705_c20_rn4.5mm.csv", 0.700)}
LABEL = {"A": "A, LV-Haack", "B": "B, Von Karman"}
STYLE = {"A": ("dark green", "-.-."), "B": ("dark blue", "----")}  # as fluent_figs.py and fig_cd_mach
D = 0.18
R = 0.09
TIP_RADIUS = 0.0045
OUTLET_X = 2.52
TM = {"burnout": (0.0723, "M 1.465 (burnout)"), "M1.20": (0.0859, "M 1.20")}  # PLAN.md section 6, Taylor-Maccoll
CFF = ["nose-drag-coefficient", "change-from-fine-mesh", "change-from-40d-domain", "relative-cell-size",
       "domain-size", "iterations-before-end"]


# ---------------------------------------------------------------- session and plots
def launch():
    # in a temporary folder, so that its transcript and cleanup script do not land in ME474
    s = pf.launch_fluent(product_version="24.2.0", dimension=2, precision="double", processor_count=4,
                         ui_mode="hidden_gui", cleanup_on_exit=True, start_timeout=300,
                         cwd=tempfile.mkdtemp(prefix="fluent_figs_study_"))
    print("Fluent started", flush=True)
    return s


def read_case(s, run):
    s.settings.file.read_case(file_name=(dc.RUNS / run / f"{run}.cas.h5").as_posix())
    for cmd in ["/display/set/windows/logo no", "/preferences/graphics/boundary-markers/enabled no"]:
        s.execute_tui(cmd)
    for name in CFF:  # constant functions: only their names show, in the plot title
        s.execute_tui(f'/define/custom-field-functions/define "{name}" "1"')


def xy_files(s, name, curves, x_func, y_func, labels, ranges, prec, out):
    """XY plot of XY files. curves: [(path, colour, pattern, symbol, weight)]. x_func None = position along x.
    ranges: ((xmin, xmax) or None for auto, same for y). With a fixed range Fluent often labels only the ends and the
    middle, so the ranges below are ones whose labels come out round; minor grid lines fill in between.
    Fluent makes the last file read curve 0, so the files are read in reverse (as in fluent_figs.file_plot)."""
    p = ff.xy_plot(s, name)
    if x_func:
        p.options.position_on_x_axis = False
        p.x_axis_function = x_func
    p.y_axis_function = y_func
    p.surfaces_list = []
    for c in reversed(curves):
        p.read_from_file(filename=str(c[0]))
    p.axes.labels = {"x_axis": labels[0], "y_axis": labels[1]}
    p.axes.numbers = {"x_format": "float", "x_axis_precision": prec[0], "y_format": "float", "y_axis_precision": prec[1]}
    p.axes.rules = {"x_axis": {"draw_major_rules": True, "draw_minor_rules": True},
                    "y_axis": {"draw_major_rules": True, "draw_minor_rules": True}}
    a = p.axes.auto_scale
    for axis, rng in zip(("x_axis", "y_axis"), ranges):
        if rng:  # 2024 R2: min and max are only settable while the flag is True, and only used once it is False again
            setattr(a, axis, True)
            setattr(a, axis + "_min", rng[0])
            setattr(a, axis + "_max", rng[1])
            setattr(a, axis, False)
    for i, (_, color, pattern, symbol, weight) in enumerate(curves):
        p.curves.lines[i] = {"pattern": pattern, "weight": weight, "color": color}
        p.curves.markers[i] = {"symbol": symbol, "size": 1.2 if symbol else 0.3, "color": color}
    p.display()
    ff.save(s, out)


def xy(tag, label, pts):
    path = ff.XY / f"{tag}.xy"
    ff.write_xy(path, label, "y", pts)
    return path


# ---------------------------------------------------------------- geometry
def profile(shape):
    """Nose r(x) from CFD/geometry_720 in m, then the coupler to x = 720 mm."""
    name, shoulder = PROFILE[shape]
    with open(GEOM / name) as fh:
        pts = [(float(r["x_mm"]) / 1000, float(r["r_mm"]) / 1000) for r in csv.DictReader(fh)]
    assert abs(pts[-1][0] - shoulder) < 1e-9 and abs(pts[-1][1] - R) < 1e-9
    return pts + [(0.720, R)]


def geometry(s):
    nose = {sh: xy(f"geom_{sh}", f"{LABEL[sh]}: nose {1000 * PROFILE[sh][1]:.0f} mm + coupler "
                   f"{720 - 1000 * PROFILE[sh][1]:.0f} mm", profile(sh)) for sh in "AB"}
    marks = [xy(f"geom_shoulder_{sh}", f"{sh} shoulder, x = {1000 * PROFILE[sh][1]:.0f} mm",
                [(PROFILE[sh][1], 0.072), (PROFILE[sh][1], 0.098)]) for sh in "AB"]
    end = xy("geom_720", "x = 720 mm: body (r = 90 mm) behind", [(0.720, 0.0), (0.720, 0.098)])
    curves = [(nose["A"], *STYLE["A"], "", 3), (nose["B"], *STYLE["B"], "", 3),
              (marks[0], STYLE["A"][0], "....", "", 2), (marks[1], STYLE["B"][0], "....", "", 2),
              (end, "dark gray", "- -", "", 2)]
    xy_files(s, "g-profiles", curves, None, "radial-coordinate", ("Position x", "Radius r"),
             (None, None), (2, 3), "fluent_geometry_profiles.png")

    n = 60
    sphere = xy("geom_tip_sphere", f"tip sphere, radius {1000 * TIP_RADIUS:.1f} mm",
                [(TIP_RADIUS * (1 - math.cos(math.pi * i / n)), TIP_RADIUS * math.sin(math.pi * i / n)) for i in range(n + 1)])
    curves = [(nose["A"], *STYLE["A"], "", 3), (nose["B"], *STYLE["B"], "", 3), (sphere, "red", "- -", "", 2)]
    # 0.035 x 0.020 m on the 2400 x 1500 picture: equal scales, so the sphere looks round
    xy_files(s, "g-tip", curves, None, "radial-coordinate", ("Position x", "Radius r"),
             ((0.0, 0.035), (0.0, 0.020)), (3, 3), "fluent_geometry_tip.png")

    body = [(0.0, 0.0)] + profile("A")[1:] + [(OUTLET_X, R)]
    curves = [(xy("dom_40", "40 D, M 0.90-1.10 (pressure far-field)",
                  [(-40 * D, 0), (-40 * D, 40 * D), (OUTLET_X, 40 * D)]), "dark blue", "----", "", 3),
              (xy("dom_20", "20 D, M 1.20, burnout, mesh study",
                  [(-20 * D, 0), (-20 * D, 20 * D), (OUTLET_X, 20 * D)]), "dark green", "- -", "", 3),
              (xy("dom_10", "10 D, domain study",
                  [(-10 * D, 0), (-10 * D, 10 * D), (OUTLET_X, 10 * D)]), "orange", "....", "", 3),
              (xy("dom_outlet", "outlet (pressure outlet)",
                  [(OUTLET_X, R), (OUTLET_X, 40 * D)]), "red", "----", "", 3),
              (xy("dom_axis", "axis", [(-40 * D, 0), (0, 0)]), "silver", "-.-.", "", 3),
              (xy("dom_wall", "wall (nose, coupler, body)", body), "dark gray", "----", "", 4)]
    xy_files(s, "g-domains", curves, None, "radial-coordinate", ("Position x", "Radius r"),
             ((-12.0, 4.0), (0.0, 8.0)), (0, 0), "fluent_geometry_domains.png")


# ---------------------------------------------------------------- mesh pictures
def wall_r(shape, x):
    pts = profile(shape)
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return r0 + (r1 - r0) * (x - x0) / (x1 - x0)
    raise ValueError(x)


def views(shape):
    """name -> camera (centre x, centre y, width, height), all 1.6 : 1 like the picture. Fluent draws the ruler at the
    bottom centre, over the lowest 6 % of the picture: each view puts that inside the body or below the axis (white)."""
    r_wall = wall_r(shape, 0.345)
    return {"nose": (0.36, 0.26, 0.88, 0.55), "tip": TIP, "tipzoom": (0.004, 0.005, 0.016, 0.01),
            "wall": (0.345, r_wall + 0.0095, 0.048, 0.03), "shoulder": (0.70, 0.10, 0.08, 0.05)}


TIP = (0.0, 0.0305, 0.112, 0.07)  # the 40 mm tip sphere; the bottom 4.5 mm (the ruler) below the axis


MESHES = [  # (run whose case holds the mesh, mesh name, views)
    ("A_D40_M0.95", "A_D40", {"domain": (-2.34, 3.45, 12.8, 8.0)}),  # bottom 0.55 m (the ruler) below the axis
    ("A_medium_M0.95", "A_medium", {"domain": (-0.54, 1.75, 6.4, 4.0), **views("A")}),
    ("B_medium_M0.95", "B_medium", views("B")),
    ("A_coarse_M0.95", "A_coarse", {k: v for k, v in views("A").items() if k in ("nose", "tip", "wall")}),
    ("A_fine_M0.95", "A_fine", {k: v for k, v in views("A").items() if k in ("nose", "tip", "wall")}),
    ("cone_burnout", "cone", {"tip": TIP, "tipzoom": (0.0, 0.00009, 0.0004, 0.00025)}),
]


def mesh_pictures(s):
    for run, mesh, vs in MESHES:
        print("===", mesh, flush=True)
        read_case(s, run)
        g = s.settings.results.graphics
        g.views.mirror_zones = []
        g.colors.foreground = "black"  # the ruler's colour; mesh pictures come last, after the XY plots
        g.windows.ruler = True
        if "mesh-all" not in g.mesh.get_object_names():
            g.mesh.create(name="mesh-all")
        m = g.mesh["mesh-all"]
        m.surfaces_list = m.surfaces_list.allowed_values()
        m.options.edges = True
        m.coloring.option = "manual"
        for color in ("black", "foreground"):  # the colour list is not exposed (allowed_values() is None)
            try:
                m.coloring.manual.edges = color
                print("mesh edges:", color, flush=True)
                break
            except Exception as e:
                print("edge colour", color, "not accepted:", str(e)[:100], flush=True)
        m.display()
        for view, cam in vs.items():
            ff.camera(s, *cam)
            ff.save(s, f"fluent_mesh_{view}_{mesh}.png")


# ---------------------------------------------------------------- mesh and domain study
def mean_last(run, n=500):
    rows = cr.read_report(dc.RUNS / run / f"{run}_cd.out")
    its = sorted(rows)[-n:]
    return sum(rows[i]["cd-nose"] for i in its) / len(its)


def mesh_study(summary, shape, point):
    """[(h/h_fine, cd)] fine, medium, coarse, and the gci.py result."""
    runs = [f"{shape}_{m}_{point}" for m in ("fine", "medium", "coarse")]
    n = [int(summary[r]["cells"]) for r in runs]
    cd = [mean_last(r) for r in runs] if point == "burnout" else [float(summary[r]["cd_nose"]) for r in runs]
    res = gci.gci(n[0], cd[0], n[1], cd[1], n[2], cd[2])
    return [(math.sqrt(n[0] / k), c) for k, c in zip(n, cd)], res


def studies(s, summary):
    ms = {(sh, pt): mesh_study(summary, sh, pt) for sh in "AB" for pt in ("M0.95", "burnout")}
    for (sh, pt), (pts, res) in ms.items():
        print(sh, pt, [(round(h, 3), round(c, 6)) for h, c in pts], "p", round(res["p"], 2), "f_ext",
              round(res["f_ext"], 6), "GCI fine %", round(res["gci_fine_pct"], 3), "oscillatory", res["oscillatory"])

    # C_D at M 0.95 against h/h_fine; 0.0312-0.0328 is a range Fluent labels at the ends and the middle
    (pa, ra), (pb, _) = ms[("A", "M0.95")], ms[("B", "M0.95")]
    f1, gfine = pa[0][1], ra["gci_fine_pct"] / 100
    curves = [(xy("ms_A_M0.95", "A, LV-Haack", pa), *STYLE["A"], "(o)", 3),
              (xy("ms_B_M0.95", "B, Von Karman (oscillatory: no GCI)", pb), *STYLE["B"], "[]", 3),
              (xy("ms_A_M0.95_rich", f"A, Richardson extrapolation to h = 0: {ra['f_ext']:.5f}",
                  [(1.0, f1), (0.0, ra["f_ext"])]), STYLE["A"][0], "....", "<>", 2),
              (xy("ms_A_M0.95_gci", f"A, GCI of the fine mesh: {100 * gfine:.2f} % either side",
                  [(1.0, f1 * (1 - gfine)), (1.0, f1 * (1 + gfine))]), "red", "----", "", 3)]
    xy_files(s, "s-mesh-M0.95", curves, "relative-cell-size", "nose-drag-coefficient",
             ("Relative cell size h/h_fine, h = (A/N)^1/2", "Cd of the nose (wall_nose), M 0.95"),
             ((0.0, 2.0), (0.0312, 0.0328)), (1, 4), "fluent_meshstudy_M0.95.png")

    # all four as change from the fine mesh
    curves = []
    for (sh, pt), symbol in ((("A", "M0.95"), "(o)"), (("B", "M0.95"), "[]"), (("A", "burnout"), "(x)"),
                             (("B", "burnout"), "[x]")):
        pts, res = ms[(sh, pt)]
        f1 = pts[0][1]
        rel = [(h, 100 * (c / f1 - 1)) for h, c in pts]
        color, pattern = STYLE[sh]
        name = f"{sh}, {'M 0.95' if pt == 'M0.95' else 'burnout'}"
        curves.append((xy(f"msc_{sh}_{pt}", name, rel), color, pattern if pt == "M0.95" else "- -", symbol, 3))
        if not res["oscillatory"]:
            curves.append((xy(f"msc_{sh}_{pt}_rich", f"{name}: Richardson",
                              [(1.0, 0.0), (0.0, 100 * (res["f_ext"] / f1 - 1))]), color, "....", "<>", 2))
    xy_files(s, "s-mesh-change", curves, "relative-cell-size", "change-from-fine-mesh",
             ("Relative cell size h/h_fine", "Cd of the nose: change from the fine mesh [%]"),
             ((0.0, 2.0), (-1.5, 1.5)), (1, 1), "fluent_meshstudy_change.png")

    # domain study: C_D against L/D
    def dom(shape, point, sizes):
        runs = {10: f"{shape}_D10_{point}", 20: f"{shape}_medium_{point}", 40: f"{shape}_D40_{point}"}
        return [(L, float(summary[runs[L]]["cd_nose"])) for L in sizes if runs[L] in summary]
    for point, rng in (("M0.95", (0.031, 0.035)), ("M1.00", (0.040, 0.060))):
        a, b = dom("A", point, (10, 20, 40)), dom("B", point, (10, 20, 40))
        curves = [(xy(f"ds_A_{point}", LABEL["A"], a), *STYLE["A"], "(o)", 3),
                  (xy(f"ds_B_{point}", LABEL["B"] + (" (40 D only)" if len(b) == 1 else ""), b), *STYLE["B"], "[]", 3)]
        xy_files(s, f"s-domain-{point}", curves, "domain-size", "nose-drag-coefficient",
                 ("Domain size L/D: inlet distance = far-field radius", f"Cd of the nose (wall_nose), M {point[1:]}"),
                 ((0.0, 50.0), rng), (0, 4), f"fluent_domainstudy_{point}.png")
    curves = []
    for sh, point, color, pattern, symbol in (("A", "M0.95", "dark green", "-.-.", "(o)"), ("B", "M0.95", "dark blue", "----", "[]"),
                                              ("A", "M1.00", "red", "-.-.", "(x)"), ("A", "M1.20", "orange", "-.-.", "<>")):
        pts = dom(sh, point, (10, 20, 40))
        ref = dict(pts)[40]
        curves.append((xy(f"dsc_{sh}_{point}", f"{LABEL[sh]}, M {point[1:]}", [(L, 100 * (c / ref - 1)) for L, c in pts]),
                       color, pattern, symbol, 3))
    xy_files(s, "s-domain-change", curves, "domain-size", "change-from-40d-domain",
             ("Domain size L/D: inlet distance = far-field radius", "Cd of the nose: change from the 40 D domain [%]"),
             ((0.0, 50.0), (-15.0, 15.0)), (0, 0), "fluent_domainstudy_change.png")


# ---------------------------------------------------------------- iterative convergence at burnout
def histories(s):
    """Last 1000 iterations of C_D (nose) on the three burnout meshes, against iterations before the end of each run
    (the coarse runs end at 8000, medium and fine at 4000), from runs/<run>/<run>_cd.out."""
    for sh, rng in (("A", (0.0932, 0.0936)), ("B", (0.0789, 0.0793))):
        curves = []
        for mesh, color, pattern in (("coarse", "red", "----"), ("medium", *STYLE[sh]), ("fine", "dark gray", "- -")):
            run = f"{sh}_{mesh}_burnout"
            rows = cr.read_report(dc.RUNS / run / f"{run}_cd.out")
            its = sorted(rows)[-1000:]
            curves.append((xy(f"hist_{run}", f"{sh} {mesh}, iterations {its[0]}-{its[-1]}",
                              [(i - its[-1], rows[i]["cd-nose"]) for i in its]), color, pattern, "", 2))
        xy_files(s, f"h-{sh}", curves, "iterations-before-end", "nose-drag-coefficient",
                 ("Iterations before the end of the run", f"Cd of the nose (wall_nose), burnout, {LABEL[sh]}"),
                 ((-1000.0, 0.0), rng), (0, 5), f"fluent_cd_history_burnout_{sh}.png")  # ticks 0.00005 apart


# ---------------------------------------------------------------- cone validation
def cone(s):
    for point, (cp_tm, mach) in TM.items():
        run = f"cone_{point}"
        with open(dc.RUNS / run / f"{run}_wall.csv") as fh:
            rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(fh, skipinitialspace=True)]
        pts = sorted((r["x-coordinate"], r["pressure-coefficient"]) for r in rows if r["x-coordinate"] <= 0.720)
        curves = [(xy(f"cone_{point}", f"Fluent, wall Cp of the sharp cone, {mach}", pts), "dark blue", "----", "", 2),
                  (xy(f"cone_tm_{point}", f"Taylor-Maccoll (inviscid): Cp = {cp_tm} (drawn over x = 0.05-0.65 m)",
                      [(0.05, cp_tm), (0.65, cp_tm)]), "red", "- -", "", 4)]
        rng = (0.06, 0.09) if point == "burnout" else (0.03, 0.13)
        xy_files(s, f"c-{point}", curves, None, "pressure-coefficient", ("Position x", "Pressure coefficient Cp"),
                 (None, rng), (2, 3), f"fluent_cp_cone_{point}.png")


def main(args):
    parts = args or ["geometry", "mesh", "studies", "history", "cone"]
    ff.FIG.mkdir(exist_ok=True)
    ff.XY.mkdir(exist_ok=True)
    summary = {r["run"]: r for r in csv.DictReader(open(dc.RUNS / "summary.csv"))}
    s = launch()
    try:
        read_case(s, "A_medium_M0.95")
        if "geometry" in parts:
            geometry(s)
        if "studies" in parts:
            studies(s, summary)
        if "history" in parts:
            histories(s)
        if "cone" in parts:
            read_case(s, "cone_burnout")
            cone(s)
        if "mesh" in parts:
            mesh_pictures(s)
    finally:
        s.exit()


if __name__ == "__main__":
    main(sys.argv[1:])

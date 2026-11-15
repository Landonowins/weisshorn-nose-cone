"""Shock angle of the validation cone runs, for the comparison with Taylor-Maccoll.

Usage: python shock_angle.py cone_burnout cone_M1.20      (Fluent session of the newest server_info*.txt here)

For each run: read its case/data, export the pressure along horizontal lines y = 10, 20, 30 mm (inside the 0.2 mm tip cells) and 150, 300, 450, 600 mm
(x = -0.1 to 1.5 m) to runs/<run>/<run>_shock_lines.csv (skipped if that file exists, unless --export), take
the shock position on each line as its foot, the first x where Cp = (p - p_inf) / q_inf reaches 0.005, and fit
the shock angle through the cone tip (x = y / tan(beta)).

Why the foot: at M 1.2-1.47 the 7.1 deg cone's shock is only 0.3 deg steeper than the Mach angle, so it is very
weak (Cp jump about 0.007) and most of the compression follows smoothly behind it; a midpoint between p_inf and
the maximum along the line lands in that compression, not at the shock. The threshold 0.005 is 65-75 % of the
theoretical jump at both Mach numbers.
"""
import csv
import json
import math
import sys
import time

import setup_fluent as F

LINES_MM = (10, 20, 30, 150, 300, 450, 600)
THEORY = {"burnout": 43.3, "M1.20": 56.7}  # Taylor-Maccoll shock angle, deg (PLAN.md step 8)


def export_lines(s, run):
    names = []
    tag = int(time.time()) % 100000  # fresh names: a surface kept from before a case read is not recomputed
    for y in LINES_MM:
        name = f"shock{tag}-y{y}"
        # a line below y = 90 mm must end before the cone surface (x = y / tan 7.125 deg), or Fluent leaves it empty
        x_end = 1.5 if y >= 90 else 0.9 * y / 1000 / 0.125
        # the settings API creates or updates; the TUI command keeps an existing surface of the same name
        s.settings.results.surfaces.line_surface[name] = {"p0": [-0.1, y / 1000, 0], "p1": [x_end, y / 1000, 0]}
        names.append(name)
    out = F.run_dir(run) / f"{run}_shock_lines.csv"
    out.unlink(missing_ok=True)  # an existing file makes Fluent ask "overwrite?", which eats the next argument
    s.tui.file.export.ascii(out.as_posix(), *names, "()", "yes", "pressure", "mach-number", "()", "no")
    return out


CP_FOOT = 0.005


def shock_positions(path, p_inf, mach):
    q = 0.5 * 1.4 * p_inf * mach ** 2
    rows = list(csv.reader(open(path)))
    head = [c.strip() for c in rows[0]]
    pts = [dict(zip(head, map(float, r))) for r in rows[1:] if r and len(r) == len(head)]
    res = {}
    for y in LINES_MM:
        line = sorted((p for p in pts if abs(p["y-coordinate"] * 1000 - y) < 1e-3), key=lambda p: p["x-coordinate"])
        cp = [((p["pressure"] - p_inf) / q, p["x-coordinate"]) for p in line]
        for (ca, xa), (cb, xb) in zip(cp, cp[1:]):
            if ca < CP_FOOT <= cb:
                res[y] = 1000 * (xa + (CP_FOOT - ca) / (cb - ca) * (xb - xa))
                break
    return res


def main(args):
    force = "--export" in args
    runs = [a for a in args if not a.startswith("--")]
    s = None
    for run in runs:
        meta = json.loads((F.run_dir(run) / "run.json").read_text())
        path = F.run_dir(run) / f"{run}_shock_lines.csv"
        if force or not path.exists():
            s = s or F.connect()
            s.settings.file.read_case_data(file_name=(F.run_dir(run) / run).as_posix())
            export_lines(s, run)
        pos = shock_positions(path, meta["p_inf"], meta["mach"])
        print(f"{run} (M {meta['mach']}, Mach angle {math.degrees(math.asin(1 / meta['mach'])):.2f} deg):")
        for y, x in pos.items():
            print(f"  y {y:4d} mm: shock foot at x {x:7.1f} mm -> {math.degrees(math.atan2(y, x)):.2f} deg")
        theory = THEORY.get(meta["point"])
        # near the tip the cells are 0.2 mm and resolve the shock; far out (10-50 mm cells) it is smeared, so its foot
        # moves upstream and the angle comes out about 1 deg too steep: report the two groups separately
        for label, group in (("near the tip, y <= 30 mm", [y for y in pos if y <= 30]),
                             ("far field, y >= 150 mm", [y for y in pos if y >= 150])):
            num = sum(pos[y] * y for y in group)
            den = sum(y * y for y in group)
            beta = math.degrees(math.atan(den / num))  # least squares x = k y through the tip, beta = atan(1/k)
            print(f"  fitted shock angle, {label}: {beta:.2f} deg"
                  + (f" (Taylor-Maccoll {theory} deg, {beta - theory:+.2f} deg)" if theory else ""))


if __name__ == "__main__":
    main(sys.argv[1:])

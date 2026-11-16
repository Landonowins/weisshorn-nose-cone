"""C_D(M) of the production runs for A and B, the drag-rise Mach number and where A and B cross.

Usage: python drag_curve.py [--field cd_nose|cd_p|cd_v]      (reads runs/summary.csv; writes runs/drag_curve.csv)

Production set (PLAN.md section 7, decided 15 Nov): 40 D (A_D40, B_D40) up to M 1.10, 20 D (A_medium, B_medium) at
M 1.20 and burnout. M 1.02 added 16 Nov, to pin down where A and B cross above M 1. A missing run is left out
and reported.

Drag rise: dC_D/dM = 0.1 (decision 3), from finite differences between neighbouring points: the slope of each interval
is put at its midpoint, and the first crossing of 0.1 is interpolated linearly between midpoints.
Crossover: where C_D,A - C_D,B changes sign between neighbouring points, interpolated linearly.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
POINTS = ("M0.90", "M0.95", "M0.98", "M1.00", "M1.02", "M1.05", "M1.10", "M1.20", "burnout")
SLOPE = 0.1


def run_name(shape, point):
    mesh = f"{shape}_D40" if point not in ("M1.20", "burnout") else f"{shape}_medium"
    return f"{mesh}_{point}"


def drag_rise(ms, cds):
    mids = [(a + b) / 2 for a, b in zip(ms, ms[1:])]
    slopes = [(d - c) / (b - a) for a, b, c, d in zip(ms, ms[1:], cds, cds[1:])]
    for (m0, s0), (m1, s1) in zip(zip(mids, slopes), zip(mids[1:], slopes[1:])):
        if s0 < SLOPE <= s1:
            return m0 + (SLOPE - s0) / (s1 - s0) * (m1 - m0), mids, slopes
    return None, mids, slopes


def main(args):
    field = args[args.index("--field") + 1] if "--field" in args else "cd_nose"
    rows = {r["run"]: r for r in csv.DictReader(open(RUNS / "summary.csv"))}
    table, missing = [], []
    for point in POINTS:
        ra, rb = rows.get(run_name("A", point)), rows.get(run_name("B", point))
        if not (ra and rb):
            missing += [run_name(s, point) for s, r in (("A", ra), ("B", rb)) if not r]
            continue
        a, b = float(ra[field]), float(rb[field])
        table.append({"point": point, "mach": float(ra["mach"]), "A": a, "B": b, "A-B": a - b,
                      "B_vs_A_pct": 100 * (b / a - 1), "run_A": ra["run"], "run_B": rb["run"]})
    with open(RUNS / "drag_curve.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    print(f"{field}:  point     M        A         B       A-B    B vs A")
    for t in table:
        print(f"        {t['point']:8s} {t['mach']:.3f}  {t['A']:.5f}  {t['B']:.5f}  {t['A-B']:+.5f}  {t['B_vs_A_pct']:+6.1f} %")
    if missing:
        print("missing:", " ".join(missing))
    ms = [t["mach"] for t in table]
    for shape in ("A", "B"):
        mdr, mids, slopes = drag_rise(ms, [t[shape] for t in table])
        print(f"{shape}: slopes " + ", ".join(f"{s:.3f} at {m:.3f}" for m, s in zip(mids, slopes)))
        print(f"   drag rise (dC_D/dM = {SLOPE}): " + (f"M = {mdr:.3f}" if mdr else "not bracketed"))
    for t0, t1 in zip(table, table[1:]):
        if t0["A-B"] * t1["A-B"] < 0:
            m = t0["mach"] + t0["A-B"] / (t0["A-B"] - t1["A-B"]) * (t1["mach"] - t0["mach"])
            print(f"A and B cross at M = {m:.3f} (between {t0['point']} and {t1['point']})")
    print(f"-> {RUNS / 'drag_curve.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])

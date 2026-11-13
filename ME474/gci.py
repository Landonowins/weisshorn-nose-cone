"""Grid Convergence Index of a quantity on three meshes (Celik et al. 2008, J. Fluids Eng. 130, 078001).

Representative cell size in 2D: h = (A / N)^(1/2), so the refinement ratio is r = (N_fine / N_coarse)^(1/2)
for meshes of the same domain. Returns the observed order p, the Richardson extrapolation f_ext and
the fine-mesh GCI (safety factor 1.25). Oscillatory convergence (the differences change sign) is
flagged: p and the GCI are then not meaningful.

Usage:
    python gci.py RUN_FINE RUN_MEDIUM RUN_COARSE [--field cd_nose]   (rows of runs/summary.csv)
    python gci.py --values N1 f1 N2 f2 N3 f3                          (fine to coarse)
"""
import csv
import math
import sys
from pathlib import Path


def gci(n1, f1, n2, f2, n3, f3, fs=1.25):
    r21, r32 = math.sqrt(n1 / n2), math.sqrt(n2 / n3)
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        raise ValueError("two meshes give the same value: the order cannot be computed")
    s = math.copysign(1, e32 / e21)
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(200):  # fixed point on p, for unequal refinement ratios
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1)
    ea32 = abs((f2 - f3) / f2)
    return dict(r21=r21, r32=r32, p=p, oscillatory=s < 0, f_ext=f_ext,
                e_a21_pct=100 * ea21, e_ext21_pct=100 * abs((f_ext - f1) / f_ext),
                gci_fine_pct=100 * fs * ea21 / (r21 ** p - 1),
                gci_medium_pct=100 * fs * ea32 / (r32 ** p - 1))


def main():
    a = sys.argv[1:]
    if a and a[0] == "--values":
        v = [float(x) for x in a[1:7]]
        res = gci(*v)
    else:
        field = a[a.index("--field") + 1] if "--field" in a else "cd_nose"
        runs = [x for x in a if not x.startswith("--") and x != field][:3]
        rows = {r["run"]: r for r in csv.DictReader(open(Path(__file__).resolve().parent / "runs" / "summary.csv"))}
        v = []
        for run in runs:
            v += [float(rows[run]["cells"]), float(rows[run][field])]
        print(f"{field}: " + ", ".join(f"{run} N {v[2 * i]:.0f} f {v[2 * i + 1]:.6g}" for i, run in enumerate(runs)))
        res = gci(*v)
    for k, x in res.items():
        print(f"  {k}: {x:.5g}" if isinstance(x, float) else f"  {k}: {x}")
    # asymptotic range: GCI_medium / (r21^p GCI_fine) close to 1
    print(f"  asymptotic ratio: {res['gci_medium_pct'] / (res['r21'] ** res['p'] * res['gci_fine_pct']):.3f}")


if __name__ == "__main__":
    main()

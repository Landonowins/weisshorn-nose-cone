"""h(x) = -q_w(T300) / (T_aw - 300 K) on wall_nose + wall_coupler, faces matched by position.

Usage: python make_h.py <adiabatic wall csv> <T300 wall csv> <out csv>
"""
import csv
import sys


def load(path):
    rows = list(csv.reader(open(path)))
    h = [c.strip() for c in rows[0]]
    return {(round(float(r[h.index("x-coordinate")]), 9), round(float(r[h.index("y-coordinate")]), 9)):
            {k: float(r[i]) for i, k in enumerate(h)} for r in rows[1:] if r}


adiab, cold = load(sys.argv[1]), load(sys.argv[2])
assert adiab.keys() == cold.keys(), "the two exports are not on the same faces"
out = []
for key in sorted(adiab, key=lambda k: k[0]):
    taw, q = adiab[key]["temperature"], -cold[key]["heat-flux"]  # Fluent: heat into the wall is negative
    out.append((key[0] * 1e3, key[1] * 1e3, taw, q, q / (taw - 300.0)))
with open(sys.argv[3], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x_mm", "r_mm", "T_aw_K", "q_w_300K_W_m2", "h_W_m2K"])
    w.writerows([f"{v:.6g}" for v in row] for row in out)
for row in out[:1] + [r for r in out if r[0] >= 10][:1] + [r for r in out if r[0] >= 100][:1] + \
        [r for r in out if r[0] >= 300][:1] + [r for r in out if r[0] >= 678][:1]:
    print("x %7.2f mm  r %6.2f mm  T_aw %6.1f K  q %8.0f W/m2  h %6.0f W/m2K" % row)

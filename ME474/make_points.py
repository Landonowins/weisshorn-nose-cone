"""Free stream of the ME-474 runs: design A's simulated ascent at the course Mach numbers, plus burnout.

Same trajectory model and inputs as the Rocket Team script CFD/study720/python_points_720.py
(studies/total_length.py with configs/weisshorn_650.yaml, design A = LV-Haack 682 + 41 mm). Design A's
points are used for both shapes, so that A against B compares shapes only. The derived values use
Fluent's default air (CFD/studyA/freestream_bc.py), so they match what Fluent computes.

Usage, from the repository root with its Python environment:
    python ME474/make_points.py              (writes ME474/fluent_points.csv)
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fluent_points.csv"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))
sys.path.insert(0, str(ROOT / "CFD" / "studyA"))

from total_length import Model  # noqa: E402
import freestream_bc as fb  # noqa: E402

MACHS = (0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20)  # + burnout; 0.98 / 1.02 are the fall-backs for M 1.00
SHAPE, L, C = "lvhaack", 0.682345026, 0.041          # design A, as python_points_720.py
L_NOSE = 0.72                                         # Reynolds length: nose with coupler [m]
G = fb.GAMMA


def pitot(p, m):
    """Total pressure on the stagnation streamline: isentropic below M 1, behind a normal shock above."""
    if m <= 1.0:
        return p * (1 + 0.5 * (G - 1) * m * m) ** (G / (G - 1))
    return p * ((G + 1) ** 2 * m * m / (4 * G * m * m - 2 * (G - 1))) ** (G / (G - 1)) \
        * (1 - G + 2 * G * m * m) / (G + 1)


mdl = Model(ROOT / "configs" / "weisshorn_650.yaml")
site = mdl.cfg.launch.site_altitude
tr = mdl.trajectory(SHAPE, L, C, detail=True)
i_max = int(np.argmax(tr["M"]))

rows = []
for mach in MACHS:  # linear in time between output steps, on the way up to burnout
    i = int(np.argmax(tr["M"][:i_max + 1] >= mach))
    w = (mach - tr["M"][i - 1]) / (tr["M"][i] - tr["M"][i - 1])
    f = lambda k: tr[k][i - 1] + w * (tr[k][i] - tr[k][i - 1])  # noqa: E731
    rows.append(("M%.2f" % mach, f("t"), f("z") + site, mach, f("p"), f("T")))
rows.append(("burnout", tr["t"][i_max], tr["z"][i_max] + site, tr["M"][i_max], tr["p"][i_max], tr["T"][i_max]))

head = ["point", "t_s", "z_m", "mach", "p_Pa", "T_K", "rho", "V_ms", "q_Pa", "Re_L", "T0_K", "p0_Pa",
        "p0_pitot_Pa", "k", "omega"]
with open(OUT, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(head)
    for point, t, z, m, p, T in rows:
        m, p, T = float(m), round(float(p), 1), round(float(T), 3)  # Fluent gets the rounded p and T
        rho = p / (fb.R_AIR * T)
        v = m * math.sqrt(G * fb.R_AIR * T)
        mu = fb.sutherland(T)
        k = 1.5 * (fb.TURB_INTENSITY * v) ** 2
        wr.writerow([point, f"{t:.3f}", f"{z:.1f}", f"{m:.5f}", f"{p:.1f}", f"{T:.3f}", f"{rho:.5f}",
                     f"{v:.2f}", f"{0.5 * rho * v * v:.0f}", f"{rho * v * L_NOSE / mu:.4e}",
                     f"{T * (1 + 0.5 * (G - 1) * m * m):.3f}", f"{p * (1 + 0.5 * (G - 1) * m * m) ** (G / (G - 1)):.0f}",
                     f"{pitot(p, m):.0f}", f"{k:.3f}", f"{rho * k / (mu * fb.TURB_VISC_RATIO):.4e}"])
print(open(OUT).read())

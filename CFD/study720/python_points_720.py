"""Python design-tool side of the 720 mm study: trajectory points and nose C_D (wave + friction).

Same model and formulas as CFD/studyA/report_figures.py (python_cd, trajectory_check), for the two
720 mm designs of 2027_C_FD_NOSE-CONE-720MM_GEOMETRY.pdf. Writes the free stream at M 0.95, 1.20 and
burnout of each design's trajectory (fluent_points_python.csv).

Usage: python CFD/study720/python_points_720.py . CFD/study720/fluent_points_python.csv
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))
sys.path.insert(0, str(ROOT / "CFD" / "studyA"))

from nosecone.flight import skin_friction  # noqa: E402
from nosecone.geometry import volume_and_area  # noqa: E402
from total_length import Model  # noqa: E402
import freestream_bc as fb  # noqa: E402

A_REF = 0.025447
DESIGNS = {  # shape, L to the sharp apex [m], coupler [m], tip-to-shoulder [m]
    "A": ("lvhaack", 0.682345026, 0.041, 0.679),
    "B": ("vonkarman", 0.70546456, 0.020, 0.700),
}

mdl = Model(ROOT / "configs" / "weisshorn_650.yaml")
site = mdl.cfg.launch.site_altitude
rough = mdl.cfg.vehicle.surface_roughness
print(f"config: R {mdl.R} m, rn {mdl.rn * 1e3:.2f} mm, site {site} m, roughness {rough}")

for name, (shape, L, c, x_sh) in DESIGNS.items():
    veh = mdl.vehicle(shape, L, c)
    _, s_wet = volume_and_area(mdl.pr(shape), L, mdl.R, mdl.rn)
    tr = mdl.trajectory(shape, L, c, detail=True)
    i_max = int(np.argmax(tr["M"]))
    print(f"\n{name} ({shape}, L {L * 1e3:.3f} + {c * 1e3:.0f} mm): nose wetted area {s_wet:.5f} m2, "
          f"apogee {float(tr['apogee']):.0f} m")
    pts = []
    for mach in (0.95, 1.20, None):
        i = i_max if mach is None else int(np.argmax(tr["M"][:i_max + 1] >= mach))
        pts.append(i)
        M, p, T = tr["M"][i], tr["p"][i], tr["T"][i]
        rho, V, mu = tr["rho"][i], tr["V"][i], tr["mu"][i]
        re_l = rho * V * x_sh / mu
        cf_s = float(skin_friction(np.array(re_l), np.array(M), x_sh, 0.0))
        cf_r = float(skin_friction(np.array(re_l), np.array(M), x_sh, rough))
        cdw = float(veh.cdw(M))
        print(f"  {'M_max' if mach is None else f'M {mach}'}: t {tr['t'][i]:.2f} s, z {tr['z'][i] + site:.0f} m, "
              f"M {M:.5f}, p {p:.0f} Pa, T {T:.2f} K | Python C_D wave {cdw:.4f}, friction smooth "
              f"{cf_s * s_wet / A_REF:.4f} rough {cf_r * s_wet / A_REF:.4f}, total(smooth) "
              f"{cdw + cf_s * s_wet / A_REF:.4f}")

# free stream at exactly M 0.95 and 1.20 (linear in time between output steps) and at M_max
import csv
rows = []
for name, (shape, L, c, x_sh) in DESIGNS.items():
    tr = mdl.trajectory(shape, L, c, detail=True)
    i_max = int(np.argmax(tr["M"]))
    for mach in (0.95, 1.20):
        i = int(np.argmax(tr["M"][:i_max + 1] >= mach))
        w = (mach - tr["M"][i - 1]) / (tr["M"][i] - tr["M"][i - 1])
        f = lambda k: tr[k][i - 1] + w * (tr[k][i] - tr[k][i - 1])  # noqa: E731
        rows.append((name, shape, "M%.2f" % mach, f("t"), f("z") + site, mach, f("p"), f("T")))
    rows.append((name, shape, "burnout", tr["t"][i_max], tr["z"][i_max] + site, tr["M"][i_max], tr["p"][i_max], tr["T"][i_max]))
out = Path(sys.argv[2])
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["design", "shape", "point", "t_s", "z_m", "mach", "p_Pa", "T_K"])
    for r in rows:
        wr.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.1f}", f"{r[5]:.5f}", f"{r[6]:.1f}", f"{r[7]:.3f}"])
print("\n" + open(out).read())

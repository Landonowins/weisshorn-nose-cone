"""Weisshorn 720 mm nose against the layout of the EuRoC winners (2027_C_FD_REPORT-5_NOSE-CONE-720MM).

Aerospace Team Graz (overall winner 2023, 2025, 2026) flies a Von Karman nose of L/D ~ 3.1 on all
three rockets (HALCYON, ISPIDA, HARPIA; measured on the CAD views of their public technical reports,
+-5 %). HALCYON carried a 3U CubeSat in a cylinder section behind that short nose. The script puts
this layout on Weisshorn and compares it with design A at the same total length, and with a short
nose that shortens the launch vehicle:

* A    : the LV-Haack design point at 720 mm (3U in the nose), from studies/final_design.py;
* G-A  : Von Karman L/D 3.1 with the 3U in the cylinder behind it (coupler from the 3U fit);
* G-B  : Von Karman L/D 3.1 with the shortest coupler, 3U not in the nose (HARPIA-like nose);
* B    : the Von Karman design point of study B, from studies/final_design.py.

Models of studies/total_length.py (report 2), imported unchanged; the rest of the vehicle behind
the joint is fixed, so a shorter nose gives a shorter vehicle.

Usage:
    python studies/compare_winners.py configs/weisshorn_720.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import final_design as fd  # noqa: E402
import total_length as tl  # noqa: E402

LD_WINNERS = 3.1           # Von Karman L/D of HALCYON, ISPIDA and HARPIA (ASTG technical reports)


def cases(mdl: tl.Model) -> list[tuple[str, str, float, float, str]]:
    """(name, shape, L, c, rule) of each compared nose [m]."""
    L_g = LD_WINNERS * 2 * mdl.R
    c_g = max(tl.C_MIN, L_g + mdl.l_grv - mdl.fit("vonkarman", L_g, 0.0)["l_in"] - L_g)  # 3U fully inside
    des = fd.design_points(mdl).query("design").set_index("study")
    a, b = des.loc["A"], des.loc["B"]
    return [("A", a["shape"], 1e-3 * a.L_mm, 1e-3 * a.c_mm, "inside"),
            ("G-A", "vonkarman", L_g, c_g, "inside"),
            ("G-B", "vonkarman", L_g, tl.C_MIN, "free"),
            ("B", b["shape"], 1e-3 * b.L_mm, 1e-3 * b.c_mm, "free")]


def evaluate(mdl: tl.Model) -> pd.DataFrame:
    rows = []
    for name, shape, L, c, rule in cases(mdl):
        ld = fd.loads(mdl, shape, L, c, rule)["nominal"]
        rows.append(dict(case=name, shape=shape, L_mm=1e3 * L, c_mm=1e3 * c, total_mm=1e3 * (L + c),
                         fineness=L / (2 * mdl.R), protrusion_mm=1e3 * mdl.fit(shape, L, c)["protrusion"],
                         apogee=mdl.apogee(shape, L, c), mass_g=1e3 * mdl.nose_mass(shape, L, c),
                         dSM=mdl.d_static_margin(shape, L, c), F_max=ld["F_max"],
                         bend_per_deg=ld["bend_per_deg"], alpha_req12=ld["alpha_req12"]))
    d = pd.DataFrame(rows)
    d["d_apogee"] = d.apogee - d.apogee.iloc[0]
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    a = ap.parse_args()
    mdl = tl.Model(Path(a.config))
    d = evaluate(mdl)
    out = Path(mdl.cfg.project.output_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "compare_winners.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Taylor-Maccoll targets of the validation cone (PLAN.md section 2h), with the repository's nosecone/taylor_maccoll.py.

Usage, with Ansys's Python (it has numpy and scipy; the VM's Python 3.14 has no scipy):
    "C:/Program Files/ANSYS/v242/commonfiles/CPython/3_10/winx64/Release/python/python.exe" ME474/taylor_maccoll_targets.py
Writes runs/taylor_maccoll_targets.csv.

The cone is the validation geometry: sharp, 720 mm long, base radius 90 mm, half-angle atan(90/720) = 7.125 deg.
Free stream: M of the burnout and M 1.20 points of fluent_points.csv; gamma = 1.4 (nosecone/constants.py).
cone_cp() integrates the Taylor-Maccoll ODE of inviscid conical flow inwards from a trial shock angle (state behind
the shock from the oblique-shock relations), finds by root finding (brentq) the shock angle whose solution meets the
cone surface, and returns the surface Cp (total-pressure loss across the shock, isentropic behind it), the shock
angle and the surface Mach number. Only nosecone/constants.py and taylor_maccoll.py are loaded, not the package
__init__, which needs the trajectory model's dependencies.
"""
import csv
import importlib.util
import math
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "nosecone"
HALF_ANGLE = math.degrees(math.atan(0.090 / 0.720))
POINTS = ("burnout", "M1.20")


def load():
    pkg = types.ModuleType("nosecone")
    pkg.__path__ = [str(PKG)]
    sys.modules["nosecone"] = pkg
    for name in ("constants", "taylor_maccoll"):
        spec = importlib.util.spec_from_file_location(f"nosecone.{name}", PKG / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nosecone.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["nosecone.taylor_maccoll"]


def main():
    tm = load()
    mach = {r["point"]: float(r["mach"]) for r in csv.DictReader(open(HERE / "fluent_points.csv"))}
    rows = []
    for point in POINTS:
        cp, beta, mc = tm.cone_cp(mach[point], HALF_ANGLE)
        rows.append([point, f"{mach[point]:.5f}", f"{HALF_ANGLE:.4f}", f"{cp:.5f}", f"{beta:.2f}", f"{mc:.4f}"])
        print(f"{point}: M {mach[point]:.5f}, cone {HALF_ANGLE:.3f} deg -> Cp {cp:.5f}, shock {beta:.2f} deg, "
              f"surface M {mc:.4f}")
    with open(HERE / "runs" / "taylor_maccoll_targets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point", "mach", "cone_half_angle_deg", "cp_surface", "shock_angle_deg", "mach_surface"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

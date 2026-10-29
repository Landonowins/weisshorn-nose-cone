"""2D axisymmetric flow domain for Fluent, as a STEP face in the XY plane (y >= 0, x = rocket axis).

Far-field rectangle minus the nose half-section, with the body running straight through to the
outlet (no base, so no wake: clean forebody numbers, as in the Euler nose-plus-cylinder setup). The
nose wall is the same interpolating spline through the profile CSV as the 3D STEP of
nosecone.export.write_step, and the boundary is split into seven edges, one per named selection:

    inlet, farfield, outlet, axis, wall_nose (curved part), wall_coupler, wall_body

Domain size from the Fluent workflow note: 20 D upstream of the tip and radially, 10 D behind the
coupler. Units mm, as the 3D STEP files.

Usage:
    python CFD/make_domain_2d.py CFD/geometry_650/nose_lvhaack_L561_c89_rn4.5mm.csv --coupler 89
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nosecone.export import _preserve_environment, _quiet_stdout  # noqa: E402


def build(csv: Path, coupler: float, out: Path, upstream: float, radial: float, downstream: float) -> list[dict]:
    import gmsh

    prof = np.loadtxt(csv, delimiter=",", skiprows=1)
    x, r = prof[:, 0], prof[:, 1]
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(r)) > 1e-9])  # as write_step
    x, r = x[keep], r[keep]
    r[0] = 0.0
    D = 2 * r[-1]
    x_sh, x_cp = x[-1], x[-1] + coupler
    x_in, x_out, y_top = -upstream * D, x_cp + downstream * D, radial * D

    with _preserve_environment(), _quiet_stdout():
        gmsh.initialize(readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("domain2d")
            occ = gmsh.model.occ
            nose = [occ.addPoint(float(a), float(b), 0.0) for a, b in zip(x, r)]
            p_cp = occ.addPoint(x_cp, r[-1], 0.0)
            p_ow = occ.addPoint(x_out, r[-1], 0.0)
            p_ot = occ.addPoint(x_out, y_top, 0.0)
            p_it = occ.addPoint(x_in, y_top, 0.0)
            p_ia = occ.addPoint(x_in, 0.0, 0.0)
            edges = {
                "axis": occ.addLine(p_ia, nose[0]),
                "wall_nose": occ.addSpline(nose),
                "wall_coupler": occ.addLine(nose[-1], p_cp),
                "wall_body": occ.addLine(p_cp, p_ow),
                "outlet": occ.addLine(p_ow, p_ot),
                "farfield": occ.addLine(p_ot, p_it),
                "inlet": occ.addLine(p_it, p_ia),
            }
            loop = occ.addCurveLoop(list(edges.values()))
            occ.addPlaneSurface([loop])
            occ.synchronize()
            gmsh.write(str(out))

            # read back: one face, seven edges, and the area against the profile integral
            gmsh.model.remove()
            gmsh.model.add("check")
            occ.importShapes(str(out))
            occ.synchronize()
            faces = gmsh.model.getEntities(2)
            curves = gmsh.model.getEntities(1)
            area = sum(occ.getMass(2, t) for _, t in faces)
            expected = (x_out - x_in) * y_top - np.trapezoid(r, x) - r[-1] * (x_out - x_sh)
            rows = []
            for _, t in curves:
                x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, t)
                rows.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, length=occ.getMass(1, t)))
        finally:
            gmsh.finalize()
    if len(faces) != 1 or len(curves) != 7:
        raise RuntimeError(f"{out.name}: {len(faces)} faces, {len(curves)} edges (expected 1 and 7)")
    print(f"{out.name}: area {area:.6e} mm2, profile integral {expected:.6e} mm2 "
          f"(rel. diff {abs(area - expected) / expected:.1e}); D = {D:g} mm, x from {x_in:g} to {x_out:g}, y to {y_top:g}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="profile CSV (x_mm, r_mm), tip to shoulder")
    ap.add_argument("--coupler", type=float, required=True, help="coupler (cylindrical part) length, mm")
    ap.add_argument("--upstream", type=float, default=20.0, help="inlet distance from the tip, in D")
    ap.add_argument("--radial", type=float, default=20.0, help="far-field height, in D")
    ap.add_argument("--downstream", type=float, default=10.0, help="outlet distance behind the coupler, in D")
    ap.add_argument("--out", type=Path, help="output STEP (default: domain2d_<csv name>.step next to the CSV)")
    a = ap.parse_args()
    out = a.out or a.csv.with_name(f"domain2d_{a.csv.stem}.step")
    for row in build(a.csv, a.coupler, out, a.upstream, a.radial, a.downstream):
        print("  edge x {x0:9.2f} .. {x1:9.2f}, y {y0:7.2f} .. {y1:7.2f} mm, length {length:9.2f} mm".format(**row))


if __name__ == "__main__":
    main()

"""Final Weisshorn nose at 720 mm total, coupler included (2027_C_FD_REPORT-5_NOSE-CONE-720MM).

ST accepted a 720 mm nose at the interface meeting of 29.10.2026 (LV = 5.40 m, SE_REQ_04). For the
design point of each study at that length (A: 3U in the nose, the baseline; B: 3U not in the nose,
the backup) the script gives:

* the design point of each profile (curved length, coupler, apogee, static-margin change);
* the profile as an equation (Haack series, tangent spherical tip, coupler cylinder) with a radius
  table, checked against the CAD points;
* the CAD (profile CSV, STL, STEP), also copied to CFD/geometry_720;
* the loads at the nose joint: axial force, bending moment and shear per degree of incidence;
* the aerodynamic heating of the Python model: peak temperatures, and the time histories that
  drive the Ansys transient-thermal model of the tip;
* the free-stream values of the Fluent heating runs at the selected trajectory points.

The models are those of studies/total_length.py (report 2), imported unchanged.

Usage:
    python studies/final_design.py configs/weisshorn_720.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import total_length as tl  # noqa: E402
from nosecone import heating as ht  # noqa: E402
from nosecone.geometry import blunt_tangent_point, radius, volume_and_area  # noqa: E402

TOTAL = 0.720              # agreed with ST on 29.10.2026, coupler included [m]
STUDY_NAMES = {"inside": "A", "free": "B"}
ALPHAS = [1.0, 2.0, 5.0]   # incidences of the joint-load table [deg]
X_TURB = [0.02, 0.05, 0.08, 0.15]   # stations of the turbulent heating history, from the tip [m]
T_WALL_REF = 300.0         # wall temperature of the isothermal Fluent runs and of h_turb here [K]
TABLE_STEP = 0.025         # step of the radius table [m]
M_FLUENT = [0.95, 1.2]     # ascent Mach numbers of the Fluent heating runs (+ max Mach, burnout)


# ------------------------------------------------------------------------------ design points
def design_points(mdl: tl.Model) -> pd.DataFrame:
    rows = []
    for rule, lim in tl.RULES.items():
        for shape in tl.PROFILES:
            c = mdl.coupler_for(shape, lim, total=TOTAL)
            L = TOTAL - c
            f = mdl.fit(shape, L, c)
            rows.append(dict(rule=rule, study=STUDY_NAMES[rule], shape=shape, c_mm=1e3 * c, L_mm=1e3 * L,
                             fineness=L / (2 * mdl.R), apogee=mdl.apogee(shape, L, c),
                             protrusion_mm=1e3 * f["protrusion"], mass_g=1e3 * mdl.nose_mass(shape, L, c),
                             dSM=mdl.d_static_margin(shape, L, c)))
    return tl.flag(pd.DataFrame(rows), mdl.cfg.flight.target_apogee)


# ------------------------------------------------------------------------------ equation
def equation(mdl: tl.Model, shape: str, L: float, c: float) -> dict:
    """Closed-form outer mould line, x measured from the blunted tip [m]."""
    pr, R, rn = mdl.pr(shape), mdl.R, mdl.rn
    x_t, x_c = blunt_tangent_point(pr, L, R, rn)
    delta = x_c - rn                                  # length removed by the blunting
    return dict(shape=shape, C=float(pr.p.get("C", float("nan"))), L=L, c=c, R=R, rn=rn,
                delta=delta, x_t=x_t - delta, r_t=float(radius(pr, x_t, L, R)), L_phys=L - delta,
                total_phys=L - delta + c)


def r_of_x(eq: dict, x: np.ndarray) -> np.ndarray:
    """Radius [m] from the equation at x [m] from the blunted tip."""
    x = np.asarray(x, dtype=float)
    rn, R, L, C = eq["rn"], eq["R"], eq["L"], eq["C"]
    sphere = np.sqrt(np.clip(rn**2 - (rn - x) ** 2, 0.0, None))
    xi = np.clip((x + eq["delta"]) / L, 0.0, 1.0)
    th = np.arccos(1.0 - 2.0 * xi)
    haack = R / math.sqrt(math.pi) * np.sqrt(np.clip(th - np.sin(2 * th) / 2 + C * np.sin(th) ** 3, 0.0, None))
    return np.where(x < eq["x_t"], sphere, np.where(x <= eq["L_phys"], haack, R))


def write_equation(eq: dict, study: str, path: Path) -> pd.DataFrame:
    """Markdown sheet with the equation, the key points and a radius table every 25 mm."""
    mm = lambda v: f"{1e3 * v:.3f}"  # noqa: E731
    xs = np.unique(np.concatenate([np.arange(0.0, eq["total_phys"], TABLE_STEP),
                                   [eq["x_t"], eq["L_phys"], eq["total_phys"]]]))
    tab = pd.DataFrame({"x_mm": 1e3 * xs, "r_mm": 1e3 * r_of_x(eq, xs)}).round(3)
    name = {"lvhaack": "LV-Haack", "vonkarman": "Von Karman"}.get(eq["shape"], eq["shape"])
    lines = [
        f"# Weisshorn nose cone, study {study}: {name} {1e3 * eq['L']:.0f} + {1e3 * eq['c']:.0f} mm",
        "",
        "x is measured along the axis from the blunted tip, r is the outer radius (mm).",
        "",
        "| Part | Range of x | r(x) |",
        "|---|---|---|",
        f"| Spherical tip | 0 to {mm(eq['x_t'])} | r = sqrt(r_n^2 - (r_n - x)^2), r_n = {mm(eq['rn'])} |",
        f"| Haack profile | {mm(eq['x_t'])} to {mm(eq['L_phys'])} | r = R/sqrt(pi) * sqrt(theta - sin(2 theta)/2 + C sin^3(theta)), "
        f"theta = arccos(1 - 2 (x + delta)/L) |",
        f"| Coupler cylinder | {mm(eq['L_phys'])} to {mm(eq['total_phys'])} | r = R |",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Haack constant C | {eq['C']:.6f} ({'LV-Haack, 1/3' if eq['C'] > 0 else 'Von Karman, 0'}) |",
        f"| Curved length to the theoretical (sharp) apex L | {mm(eq['L'])} mm |",
        f"| Body radius R | {mm(eq['R'])} mm (D = {2e3 * eq['R']:.0f} mm) |",
        f"| Tip radius r_n | {mm(eq['rn'])} mm |",
        f"| Length removed by the blunting delta = x_c - r_n | {mm(eq['delta'])} mm |",
        f"| Tangency point (from the blunted tip) | x = {mm(eq['x_t'])} mm, r = {mm(eq['r_t'])} mm |",
        f"| Curved part, blunted tip to shoulder | {mm(eq['L_phys'])} mm |",
        f"| Coupler cylinder c | {mm(eq['c'])} mm |",
        f"| Total physical length | {mm(eq['total_phys'])} mm |",
        "",
        "The profile is tangent to the cylinder at the shoulder (dr/dx = 0) and to the sphere at the",
        "tangency point.",
        "",
        "| x [mm] | r [mm] |",
        "|---:|---:|",
    ] + [f"| {a:.3f} | {b:.3f} |" for a, b in zip(tab.x_mm, tab.r_mm)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tab


def check_equation(eq: dict, csv: Path) -> float:
    """Largest |r_equation - r_CAD| [mm] over the points of the profile CSV."""
    d = pd.read_csv(csv)
    return float(np.max(np.abs(1e3 * r_of_x(eq, 1e-3 * d.x_mm.values) - d.r_mm.values)))


# ------------------------------------------------------------------------------ loads
def loads(mdl: tl.Model, shape: str, L: float, c: float, rule: str) -> dict:
    """Joint loads of report 2 (nominal and hot-day envelope), plus shear and bending at incidence."""
    m_pay = tl.CANSAT_MASS + (tl.GRV_MASS if rule == "inside" else 0.0)
    out = {}
    for case, env in (("nominal", False), ("envelope", True)):
        tr = mdl.trajectory(shape, L, c, envelope=env, detail=True)
        ld = mdl.joint_loads(shape, L, c, tr, env, m_pay)
        ld["shear_per_deg"] = ld["bend_per_deg"] / ld["arm"]
        ld["alpha_req12"] = tl.REQ12_NM / ld["bend_per_deg"]
        ld["M_max"] = float(tr["M"].max())
        out[case] = ld
    out["m_payload"] = m_pay
    return out


# ------------------------------------------------------------------------------ heating
def heating_history(mdl: tl.Model, shape: str, L: float, tr: dict) -> pd.DataFrame:
    """Time histories of the Python heating model (boundary conditions of the Ansys model)."""
    h, cfg = mdl.cfg.heating, mdl.cfg
    mats = mdl.st.materials()
    pr = mdl.pr(shape)
    t, M, p, T, V = tr["t"], tr["M"], tr["p"], tr["T"], tr["V"]
    d = dict(t=t, z=tr["z"], M=M, V=V, p=p, T=T)
    hs, T0 = zip(*(ht.h_stagnation(M[i], p[i], T[i], mdl.rn) for i in range(len(t))))
    d["h_stag"], d["T0"] = np.array(hs), np.array(T0)
    for x in X_TURB:
        hx, Tr = zip(*(ht.h_turbulent(M[i], p[i], T[i], max(V[i], 1e-3), x, T_WALL_REF) for i in range(len(t))))
        d[f"h_turb_{1e3 * x:.0f}mm"], d[f"T_r_{1e3 * x:.0f}mm"] = np.array(hx), np.array(Tr)
    tip = ht.tip_insert(tr, pr, L, mdl.R, mdl.rn, h.tip_length, cfg.structure.tip_mass, mats[h.tip_material])
    d["T_tip_insert"] = np.concatenate([[T[0]], tip["T"]])
    wr = ht.wall_response(tr, mats[h.shell_material], h.shell_thickness, 0.05, n=h.wall_nodes)
    d["T_shell_out_50mm"] = np.concatenate([[T[0]], wr["T_out"]])
    d["T_shell_in_50mm"] = np.concatenate([[T[0]], wr["T_in"]])
    return pd.DataFrame(d)


def fluent_points(mdl: tl.Model, hist: pd.DataFrame) -> pd.DataFrame:
    """Free stream of the Fluent heating runs: ascent Mach numbers, burnout / maximum Mach, coast."""
    e = mdl.cfg.engine
    i_max = int(hist.M.idxmax())
    up, down = hist.loc[:i_max], hist.loc[i_max:]
    picks = {f"ascent M{m:g}": int((up.M - m).abs().idxmin()) for m in M_FLUENT}
    picks["burnout"] = int((hist.t - e.burn_time).abs().idxmin())
    picks["M_max"] = i_max
    picks["q_stag_max"] = int((hist.h_stag * (hist.T0 - T_WALL_REF)).idxmax())
    picks[f"coast M{max(M_FLUENT):g}"] = int((down.M - max(M_FLUENT)).abs().idxmin())
    by_index = {}
    for name, i in picks.items():                       # one row per instant, names merged
        by_index[i] = f"{by_index[i]} = {name}" if i in by_index else name
    rows = []
    for i, name in by_index.items():
        r = hist.loc[i]
        p0, T0 = ht.stagnation_state(r.M, r.p, r["T"])   # r.T would be the transpose
        rho = r.p / (tl.NC.R_AIR * r["T"])
        rows.append(dict(point=name, t=r.t, z=r.z, M=r.M, V=r.V, p=r.p, T=r["T"], rho=rho,
                         mu=float(ht.sutherland(r["T"])), Re_per_m=rho * r.V / float(ht.sutherland(r["T"])),
                         p0=float(p0), T0=float(T0), h_stag_python=r.h_stag,
                         q_stag_python=r.h_stag * (r.T0 - T_WALL_REF)))
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--cad-copy", default="CFD/geometry_720", help="folder that also receives the CAD files")
    ap.add_argument("--cfd-dir", default="CFD/study720", help="folder for the Fluent and Ansys inputs")
    a = ap.parse_args()
    mdl = tl.Model(Path(a.config))
    out = Path(mdl.cfg.project.output_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    cfd = ROOT / a.cfd_dir
    cfd.mkdir(parents=True, exist_ok=True)

    d = design_points(mdl)
    d.to_csv(out / "design_points.csv", index=False)
    des = d[d.design]
    cad = tl.export_cad(mdl, des, out / "cad")
    dst = ROOT / a.cad_copy
    dst.mkdir(parents=True, exist_ok=True)
    for cc in cad.values():
        for ext in ("step", "stl", "csv"):
            src = out / "cad" / f"{cc['stem']}.{ext}"
            if src.exists():
                shutil.copy2(src, dst / src.name)

    summary = dict(total_mm=1e3 * TOTAL, k=mdl.st.k, design_points=d.to_dict("records"), cad=cad,
                   equations={}, loads={}, heating={}, fluent_points={})
    for _, r in des.iterrows():
        s, L, c, study = r["shape"], 1e-3 * r.L_mm, 1e-3 * r.c_mm, r.study
        eq = equation(mdl, s, L, c)
        stem = cad[f"{r.rule}:{s}"]["stem"]
        tab = write_equation(eq, study, out / f"equation_{study}.md")
        tab.to_csv(out / f"radius_table_{study}.csv", index=False)
        shutil.copy2(out / f"equation_{study}.md", dst / f"equation_{study}_{stem}.md")
        eq["max_dev_vs_cad_mm"] = check_equation(eq, out / "cad" / f"{stem}.csv")
        summary["equations"][study] = eq
        summary["loads"][study] = loads(mdl, s, L, c, r.rule)
        nom = mdl.trajectory(s, L, c, detail=True)
        env = mdl.trajectory(s, L, c, envelope=True, detail=True)
        summary["heating"][study] = mdl.heating(s, L, nom, env)
        for case, tr in (("nominal", nom), ("envelope", env)):
            hist = heating_history(mdl, s, L, tr)
            hist.to_csv(out / f"heating_history_{study}_{case}.csv", index=False)
            if study == "A":
                hist.to_csv(cfd / f"heating_history_A_{case}.csv", index=False)
                if case == "nominal":
                    fp = fluent_points(mdl, hist)
                    fp.to_csv(out / "fluent_points_A.csv", index=False)
                    fp.to_csv(cfd / "fluent_points_A.csv", index=False)
                    summary["fluent_points"] = fp.to_dict("records")

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("equations", "loads", "heating")}, indent=1, default=float))
    print(pd.DataFrame(summary["fluent_points"]).round(4).to_string(index=False))
    print(f"outputs in {out}, CAD in {dst}, Fluent/Ansys inputs in {cfd}")


if __name__ == "__main__":
    main()

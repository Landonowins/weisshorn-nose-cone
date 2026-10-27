"""Figures and the Python comparison table of report 4 (Fluent study A, LV-Haack 561 + 89 mm).

* fig_cd_mach: pressure (wave), viscous and total C_D of the nose against Mach, from the Fluent
  forces (Fluent cannot plot across separate runs; every other result plot is Fluent's own).
* fig_python_compare: the design tool's nose wave drag and flat-plate friction against Mach,
  with the Fluent pressure and viscous parts on top.
* tab_python_compare.tex: the same numbers as a table.
The script also integrates the wall_nose face values of wall_data/*.csv back into the pressure and
viscous C_D, as a check of the export.

The Python values are those of report 2 for this nose: the wave-drag model of the design tool
(sharp-tip Euler database, C_D f^2 interpolated in L/D, transonic ramp below M = 1.1), as used in
the report-2 trajectory, and the tool's turbulent flat-plate friction (Schlichting with its
compressibility correction) applied to the nose alone. The friction is given for a smooth wall,
as in Fluent, and for the 20 um roughness of the trajectory model. The script also prints the
free stream of the report-2 trajectory of this nose at the three Mach numbers, to compare with the
trajectory table used for the Fluent runs.

Usage:
    python CFD/studyA/report_figures.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from nosecone import config as cfgmod  # noqa: E402
from nosecone.flight import skin_friction  # noqa: E402
from nosecone.geometry import volume_and_area  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402

import freestream_bc as fb  # noqa: E402

CONFIG = ROOT / "configs" / "weisshorn_650.yaml"
FIGDIR = ROOT / "CFD" / "source" / "figures"
TABLE = ROOT / "CFD" / "source" / "sections" / "tab_python_compare.tex"

PROFILE, L_CURVED, COUPLER = "lvhaack", 0.561, 0.089
RN = 0.0045                         # tip radius [m]

WALL_FILES = {0.95: "wall_nose_M0.95.csv", 1.20: "wall_nose_M1.20.csv", 1.47: "wall_nose_M1.47.csv"}


def wall_data(mach: float) -> pd.DataFrame:
    """wall_nose face values of one run, sorted from the tip to the shoulder."""
    d = pd.read_csv(HERE / "wall_data" / WALL_FILES[mach], skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    return d.sort_values("x-coordinate").reset_index(drop=True)


def integrate_wall(d: pd.DataFrame, p_inf: float, q: float) -> tuple[float, float]:
    """Pressure and viscous C_D from the face values, on the surface of revolution.

    Face ends are taken halfway between neighbouring centroids (tip and shoulder at the ends);
    F_p = sum (p - p_inf) 2 pi y dy, F_v = sum tau_w 2 pi y dx (shear pointing downstream).
    """
    x, y = d["x-coordinate"].to_numpy(), d["y-coordinate"].to_numpy()
    xe = np.concatenate(([0.0], 0.5 * (x[1:] + x[:-1]), [fb.X_SHOULDER]))
    ye = np.concatenate(([0.0], 0.5 * (y[1:] + y[:-1]), [0.5 * fb.D_REF]))
    ring = 2 * np.pi * y
    f_p = np.sum((d["pressure"].to_numpy() - p_inf) * ring * np.diff(ye))
    f_v = np.sum(d["wall-shear"].to_numpy() * ring * np.diff(xe))
    return f_p / (q * fb.A_REF), f_v / (q * fb.A_REF)


def fluent_cd() -> pd.DataFrame:
    """Free stream and wall_nose C_D of the three runs, recomputed from the Fluent forces."""
    rows = []
    for mach, alt, p, t, f_p, f_v in fb.CASES:
        rho = p / (fb.R_AIR * t)
        v = mach * math.sqrt(fb.GAMMA * fb.R_AIR * t)
        mu = fb.sutherland(t)
        q = 0.5 * rho * v * v
        rows.append(dict(mach=mach, alt=alt, p=p, T=t, rho=rho, V=v, mu=mu, q=q,
                         cd_p=f_p / (q * fb.A_REF), cd_v=f_v / (q * fb.A_REF)))
    d = pd.DataFrame(rows)
    d["cd"] = d.cd_p + d.cd_v
    return d


def python_model():
    """Wave-drag model and nose wetted area of the design tool for this nose."""
    from total_length import Model
    mdl = Model(CONFIG)
    veh = mdl.vehicle(PROFILE, L_CURVED, COUPLER)
    _, s_wet = volume_and_area(mdl.pr(PROFILE), L_CURVED, mdl.R, RN)
    return mdl, veh, s_wet


def python_cd(mdl, veh, s_wet: float, fl: pd.DataFrame) -> pd.DataFrame:
    """Python wave drag and nose friction at the Fluent free stream."""
    rough = mdl.cfg.vehicle.surface_roughness
    out = []
    for r in fl.itertuples():
        re_l = r.rho * r.V * fb.X_SHOULDER / r.mu
        cf_s = float(skin_friction(np.array(re_l), np.array(r.mach), fb.X_SHOULDER, 0.0))
        cf_r = float(skin_friction(np.array(re_l), np.array(r.mach), fb.X_SHOULDER, rough))
        out.append(dict(mach=r.mach, cd_w=float(veh.cdw(r.mach)), cd_f=cf_s * s_wet / fb.A_REF,
                        cd_f_rough=cf_r * s_wet / fb.A_REF))
    return pd.DataFrame(out)


def boost(mdl) -> dict:
    """Report-2 trajectory of this nose from lift-off to the maximum Mach number."""
    tr = mdl.trajectory(PROFILE, L_CURVED, COUPLER, detail=True)
    i_max = int(np.argmax(tr["M"]))
    return {k: v[:i_max + 1] for k, v in tr.items() if isinstance(v, np.ndarray)}


def trajectory_check(mdl, tr: dict) -> None:
    """Free stream of the report-2 trajectory of this nose where it passes the three Mach numbers."""
    site = mdl.cfg.launch.site_altitude
    print(f"\nreport-2 trajectory of this nose: M_max = {tr['M'][-1]:.3f} at {tr['z'][-1] + site:.0f} m, "
          f"p {tr['p'][-1]:.0f} Pa, T {tr['T'][-1]:.1f} K")
    for mach, alt, p, t, *_ in fb.CASES:
        if mach > tr["M"][-1]:
            print(f"  M {mach}: not reached by this nose (table {alt} m, {p:.0f} Pa, {t} K)")
            continue
        i = int(np.argmax(tr["M"] >= mach))
        print(f"  M {mach}: altitude {tr['z'][i] + site:.0f} m (table {alt} m), "
              f"p {tr['p'][i]:.0f} Pa (table {p:.0f}), T {tr['T'][i]:.1f} K (table {t})")


def friction_along(mdl, s_wet: float, tr: dict, roughness: float) -> tuple:
    """Python nose friction C_D along the boost, against Mach."""
    ok = tr["M"] >= 0.8
    m = tr["M"][ok]
    re_l = tr["rho"][ok] * tr["V"][ok] * fb.X_SHOULDER / tr["mu"][ok]
    return m, skin_friction(re_l, m, fb.X_SHOULDER, roughness) * s_wet / fb.A_REF


def panel_label(ax, letter: str) -> None:
    """Panel label (a), (b), ... in the top right corner, which these plots leave free."""
    ax.text(0.97, 0.95, fs.tex(rf"\textbf{{({letter})}}"), transform=ax.transAxes, ha="right", va="top",
            color=fs.INK)


def fig_cd_mach(fl: pd.DataFrame, save) -> None:
    """Nose C_D against Mach: pressure (wave), viscous and total, from the Fluent forces."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.42 * fs.WIDTH))
    for col, label, color, mk in (("cd", "total", fs.INK, "o"), ("cd_p", "pressure (wave)", "#2a78d6", "s"),
                                  ("cd_v", "viscous (friction)", "#eb6834", "^")):
        # dotted: only the three runs are known, not the curve between them (transonic rise)
        ax.plot(fl.mach, fl[col], color=color, marker=mk, ms=5, lw=0.8, ls=(0, (1, 2)), label=label)
        for m, v in zip(fl.mach, fl[col]):
            left = m < 1  # the three M = 0.95 values are close together: label them on the left
            ax.annotate(f"{v:.4f}", (m, v), textcoords="offset points", fontsize=7, color=color,
                        xytext=(-7, {"cd": 5, "cd_v": 0, "cd_p": 5}[col]) if left else (7, 4),
                        ha="right" if left else "left", va="center" if left else "bottom")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$C_D$ of the nose ($A_{\rm ref} = \pi D^2/4$)")
    ax.set_xlim(0.8, 1.6)
    ax.set_ylim(0, 0.14)
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig_cd_mach")


def fig_compare(mdl, veh, s_wet: float, tr: dict, fl: pd.DataFrame, save) -> None:
    import matplotlib.pyplot as plt
    m = np.linspace(0.8, 1.6, 321)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.4 * fs.WIDTH))
    a.plot(m, veh.cdw(m), color=fs.INK, label="Python (report 2)")
    a.plot(fl.mach, fl.cd_p, "o", color="#2a78d6", ms=5, label="Fluent, pressure part")
    a.set_ylabel(r"pressure (wave) $C_D$ of the nose")
    a.set_ylim(0, 0.11)
    a.annotate("transonic ramp of the model\n(no Euler data below $M = 1.1$)", xy=(0.93, float(veh.cdw(0.93))),
               xytext=(1.06, 0.050), fontsize=7, color=fs.INK2, va="center",
               arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.6))
    # friction along the boost of the report-2 trajectory (Re and M change together)
    b.plot(*friction_along(mdl, s_wet, tr, 0.0), "-", color=fs.INK, label="Python flat plate, smooth")
    b.plot(*friction_along(mdl, s_wet, tr, mdl.cfg.vehicle.surface_roughness), "--", color=fs.INK2,
           label=r"Python flat plate, \SI{20}{\micro m} rough")
    b.plot(fl.mach, fl.cd_v, "s", color="#eb6834", ms=5, label="Fluent, viscous part")
    b.set_ylabel(r"friction (viscous) $C_D$ of the nose")
    b.set_ylim(0, 0.04)
    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_xlabel(r"$M$")
        ax.set_xlim(0.8, 1.6)
        panel_label(ax, letter)
    a.legend(loc="lower right", frameon=True, edgecolor=fs.GRID, fancybox=False)
    b.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_python_compare")


def table(fl: pd.DataFrame, py: pd.DataFrame) -> str:
    def pct(a, b):
        return f"${(a / b - 1) * 100:+.1f}$\\,\\%"
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{pressure (wave) $C_D$} & \multicolumn{3}{c}{friction (viscous) $C_D$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"$M$ & Python & Fluent & difference & Python & Fluent & difference \\", r"\midrule"]
    for f, p in zip(fl.itertuples(), py.itertuples()):
        lines.append(f"{f.mach:g} & {p.cd_w:.4f} & {f.cd_p:.4f} & {pct(f.cd_p, p.cd_w)} & "
                     f"{p.cd_f:.4f} & {f.cd_v:.4f} & {pct(f.cd_v, p.cd_f)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    fs.setup()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        import matplotlib.pyplot as plt
        for ext in ("pdf", "png"):
            fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)

    fl = fluent_cd()
    mdl, veh, s_wet = python_model()
    py = python_cd(mdl, veh, s_wet, fl)
    print(f"nose wetted area {s_wet:.5f} m2 (S/A_ref = {s_wet / fb.A_REF:.3f}), "
          f"Python wave C_D at M 1.1 = {float(veh.cdw(1.1)):.4f}")
    print(pd.concat([fl[["mach", "cd_p", "cd_v", "cd"]], py.drop(columns="mach")], axis=1)
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    tr = boost(mdl)
    trajectory_check(mdl, tr)

    print("\nC_D integrated from wall_data (pressure, viscous) against the Fluent forces:")
    for r in fl.itertuples():
        cd_p, cd_v = integrate_wall(wall_data(r.mach), r.p, r.q)
        print(f"  M {r.mach}: {cd_p:.5f} / {cd_v:.5f}  (Fluent {r.cd_p:.5f} / {r.cd_v:.5f})")

    TABLE.write_text(table(fl, py), encoding="utf-8")
    fig_cd_mach(fl, save)
    fig_compare(mdl, veh, s_wet, tr, fl, save)
    print(f"\nwrote {TABLE.relative_to(ROOT)} and the figures in {FIGDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

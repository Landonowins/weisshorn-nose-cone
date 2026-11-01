"""C_D against Mach of the 720 mm noses, Fluent against the Python design tool: fig_cd_mach and
fig_python_compare of REPORT-4 (CFD/studyA/report_figures.py), here for designs A and B together.

Fluent: wall_nose forces of each converged run (fluent_cd_720.csv, from Fluent's wall-forces report).
Python: the design tool's wave-drag model (sharp-tip Euler database, transonic ramp below M = 1.1)
and flat-plate friction along the boost of each design's own trajectory, as in report 2.

Usage: python report_figures_720.py <checkout of main with nosecone/, studies/, configs/, data/>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from nosecone.flight import skin_friction  # noqa: E402
from nosecone.geometry import volume_and_area  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402
from total_length import Model  # noqa: E402

A_REF = 0.025447
FIG = HERE / "figures"
# shape, L to the sharp apex [m], coupler [m], tip to shoulder [m], label
DESIGNS = {"A": ("lvhaack", 0.682345026, 0.041, 0.679, "A, LV-Haack"),
           "B": ("vonkarman", 0.70546456, 0.020, 0.700, "B, Von Kármán")}
STYLE = {"A": dict(ls=(0, (1, 2)), mfc=None), "B": dict(ls="--", mfc="white")}  # B: open markers
C_TOT, C_P, C_V = fs.INK, "#2a78d6", "#eb6834"


def model():
    mdl = Model(ROOT / "configs" / "weisshorn_650.yaml")
    out = {}
    for d, (shape, L, c, x_sh, _) in DESIGNS.items():
        _, s_wet = volume_and_area(mdl.pr(shape), L, mdl.R, mdl.rn)
        tr = mdl.trajectory(shape, L, c, detail=True)
        i_max = int(np.argmax(tr["M"]))
        tr = {k: v[:i_max + 1] for k, v in tr.items() if isinstance(v, np.ndarray)}
        out[d] = dict(veh=mdl.vehicle(shape, L, c), s_wet=s_wet, tr=tr, x_sh=x_sh)
    return mdl, out


def friction_along(m, roughness):
    tr, ok = m["tr"], m["tr"]["M"] >= 0.8
    re_l = tr["rho"][ok] * tr["V"][ok] * m["x_sh"] / tr["mu"][ok]
    return tr["M"][ok], skin_friction(re_l, tr["M"][ok], m["x_sh"], roughness) * m["s_wet"] / A_REF


def python_at(m, mach):
    """Python wave and smooth flat-plate friction C_D at the trajectory point of this Mach number."""
    tr = m["tr"]
    i = int(np.argmin(abs(tr["M"] - mach)))
    re_l = tr["rho"][i] * tr["V"][i] * m["x_sh"] / tr["mu"][i]
    cf = float(skin_friction(np.array(re_l), np.array(mach), m["x_sh"], 0.0))
    return float(m["veh"].cdw(mach)), cf * m["s_wet"] / A_REF


def fig_cd_mach(fl, save):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.42 * fs.WIDTH))
    for d in DESIGNS:
        f = fl[fl.design == d].sort_values("mach")
        for col, name, color, mk in (("cd", "total", C_TOT, "o"), ("cd_p", "pressure (wave)", C_P, "s"),
                                     ("cd_v", "viscous (friction)", C_V, "^")):
            st = STYLE[d]
            ax.plot(f.mach, f[col], color=color, marker=mk, ms=5, lw=0.8, ls=st["ls"],
                    mfc=st["mfc"] or color, label=f"{d}: {name}")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$C_D$ of the nose ($A_{\rm ref} = \pi D^2/4$)")
    ax.set_xlim(0.8, 1.6)
    ax.set_ylim(0, 0.15)
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_cd_mach_720")


def fig_compare(mdl, mods, fl, save):
    import matplotlib.pyplot as plt
    m = np.linspace(0.8, 1.6, 321)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.4 * fs.WIDTH))
    for d, (_, _, _, _, label) in DESIGNS.items():
        f, md, ls = fl[fl.design == d], mods[d], "-" if d == "A" else "--"
        a.plot(m, md["veh"].cdw(m), ls, color=fs.INK, lw=1.0, label=f"Python, {label}")
        a.plot(f.mach, f.cd_p, "o", color=C_P, ms=5, mfc=STYLE[d]["mfc"] or C_P, label=f"Fluent, {label}")
        b.plot(*friction_along(md, 0.0), ls, color=fs.INK, lw=1.0, label=f"Python flat plate, {label}")
        b.plot(f.mach, f.cd_v, "s", color=C_V, ms=5, mfc=STYLE[d]["mfc"] or C_V, label=f"Fluent, {label}")
    a.set_ylabel(r"pressure (wave) $C_D$ of the nose")
    a.set_ylim(0, 0.10)
    va = mods["A"]["veh"]
    a.annotate("transonic ramp of the model\n(no Euler data below $M = 1.1$)", xy=(0.93, float(va.cdw(0.93))),
               xytext=(0.82, 0.088), fontsize=7, color=fs.INK2, va="center",
               arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.6))
    b.set_ylabel(r"friction (viscous) $C_D$ of the nose")
    b.set_ylim(0, 0.04)
    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_xlabel(r"$M$")
        ax.set_xlim(0.8, 1.6)
        ax.text(0.97, 0.95, f"({letter})", transform=ax.transAxes, ha="right", va="top", color=fs.INK,
                fontweight="bold")
    a.legend(loc="lower right", fontsize=6.5, frameon=True, edgecolor=fs.GRID, fancybox=False)
    b.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_python_compare_720")


def main():
    fs.setup()
    FIG.mkdir(exist_ok=True)

    def save(fig, name):
        import matplotlib.pyplot as plt
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)
        print("saved", name)

    fl = pd.read_csv(HERE / "fluent_cd_720.csv")
    mdl, mods = model()
    rows = []
    for r in fl.itertuples():
        w, f = python_at(mods[r.design], r.mach)
        rows.append(dict(run=r.run, mach=r.mach, fluent_p=r.cd_p, python_w=w, dp_pct=100 * (r.cd_p / w - 1),
                         fluent_v=r.cd_v, python_f=f, dv_pct=100 * (r.cd_v / f - 1),
                         fluent=r.cd, python=w + f, d_pct=100 * (r.cd / (w + f) - 1)))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab.to_csv(HERE / "python_compare_720.csv", index=False, float_format="%.6f")
    fig_cd_mach(fl, save)
    fig_compare(mdl, mods, fl, save)


if __name__ == "__main__":
    main()

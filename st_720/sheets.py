"""Sheets for ST of the 720 mm Weisshorn nose: geometry for SolidWorks, and joint loads / heating.

Both read the outputs of studies/final_design.py (run it first) and compile with latexmk:

* 2027_C_FD_NOSE-CONE-720MM_GEOMETRY.pdf: outer mould line of designs A and B as equations, the
  same profile as a SolidWorks equation-driven curve (parametric, text to paste), the sketch steps
  and a radius table; a SolidWorks XYZ point file next to the CAD in CFD/geometry_720;
* 2027_C_FD_NOSE-CONE-720MM_LOADS-THERMAL.pdf: loads at the nose joint (axial, bending, shear,
  limit and ultimate) and temperatures of the Python heating model, against the ERT requirements.

Usage:
    python st_720/sheets.py configs/weisshorn_720.yaml [--only geometry|loads]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import total_length as tl  # noqa: E402
from nosecone.heating import stagnation_state  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
CAD_DIR = ROOT / "CFD" / "geometry_720"
GEO_NAME = "2027_C_FD_NOSE-CONE-720MM_GEOMETRY"
LOADS_NAME = "2027_C_FD_NOSE-CONE-720MM_LOADS-THERMAL"
AUTHOR = "Fabio Feig (FD)"
TITLES = {"A": "LV-Haack, 3U payload inside the nose (baseline)",
          "B": "Von K\\'arm\\'an, 3U payload not in the nose (backup)"}
NAMES = {"lvhaack": "LV-Haack", "vonkarman": "Von K\\'arm\\'an"}

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=15mm,top=12mm,bottom=14mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,graphicx,booktabs,fancyvrb,xcolor,siunitx,array}
\sisetup{group-minimum-digits=4}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{8pt}{3pt}
\titlespacing*{\subsection}{0pt}{6pt}{2pt}
\pagestyle{plain}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
"""


def header(title: str, doc_id: str, date: str) -> str:
    return (rf"\noindent{{\Large\textbf{{{title}}}}}\\[2pt]" "\n"
            rf"{doc_id.replace('_', chr(92) + '_')} \hfill {AUTHOR}, {date}" "\n"
            r"\par\medskip\hrule\medskip" "\n")


def compile_tex(name: str, tex: str) -> Path:
    (HERE / f"{name}.tex").write_text(tex, encoding="utf-8")
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                   cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    return HERE / f"{name}.pdf"


# ============================================================================== geometry
def solidworks(eq: dict) -> dict:
    """Parametric equation-driven curve of the Haack part in SolidWorks syntax [mm, rad].

    x(t) = L/2 (1 - cos t) - delta, y(t) = R/sqrt(pi) sqrt(t - sin(2t)/2 + C sin^3 t), t in [theta_T, pi];
    theta_T is the angle of the tangency point with the spherical tip.
    """
    L, d, R, C = 1e3 * eq["L"], 1e3 * eq["delta"], 1e3 * eq["R"], eq["C"]
    th_T = math.acos(1.0 - 2.0 * (eq["x_t"] + eq["delta"]) / eq["L"])
    tail = f" + {C:.7f}*sin(t)^3" if C else ""
    return dict(x=f"{L / 2:.6f}*(1 - cos(t)) - {d:.6f}",
                y=f"{R / math.sqrt(math.pi):.6f}*sqr(t - sin(2*t)/2{tail})",
                t1=f"{th_T:.7f}", t2="pi", theta_T=th_T)


def eval_solidworks(expr: str, t: np.ndarray) -> np.ndarray:
    """Evaluate a SolidWorks expression of t (sqr = square root, ^ = power)."""
    code = expr.replace("sqr(", "np.sqrt(").replace("sin(", "np.sin(").replace("cos(", "np.cos(").replace("^", "**")
    return eval(code, {"np": np, "pi": math.pi, "t": t})  # noqa: S307 (our own generated text)


def check_solidworks(eq: dict, sw: dict, csv: Path) -> dict:
    """The pasted curve against the CAD points and the joints with the arc and the coupler [mm]."""
    t = np.linspace(sw["theta_T"], math.pi, 4001)
    x, y = eval_solidworks(sw["x"], t), eval_solidworks(sw["y"], t)
    d = pd.read_csv(csv)
    d = d[d.x_mm >= 1e3 * eq["x_t"]]
    dev = np.max(np.abs(np.interp(d.x_mm, x, y) - d.r_mm))
    slope_curve = (y[1] - y[0]) / (x[1] - x[0])
    xt, rt, rn = 1e3 * eq["x_t"], 1e3 * eq["r_t"], 1e3 * eq["rn"]
    slope_arc = (rn - xt) / rt                         # circle centred at (rn, 0)
    return dict(max_dev_mm=float(dev), start_gap_mm=float(math.hypot(x[0] - xt, y[0] - rt)),
                end=(float(x[-1]), float(y[-1])), slope_jump=float(abs(slope_curve - slope_arc)),
                end_slope=float((y[-1] - y[-2]) / (x[-1] - x[-2])))


def write_xyz(csv: Path, out: Path) -> None:
    """SolidWorks 'Curve Through XYZ Points' file: tip to shoulder, X along the axis, Y = r, Z = 0 [mm]."""
    d = pd.read_csv(csv)
    out.write_text("".join(f"{x:.4f}\t{r:.4f}\t0.0000\n" for x, r in zip(d.x_mm, d.r_mm)), encoding="utf-8")


def fig_profile(study: str, eq: dict, csv: Path, path: Path) -> None:
    d = pd.read_csv(csv)
    xs, tot, R = 1e3 * eq["L_phys"], 1e3 * eq["total_phys"], 1e3 * eq["R"]
    x = np.concatenate([d.x_mm, [tot]])
    r = np.concatenate([d.r_mm, [R]])
    col = fs.COLORS[eq["shape"]]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.25 * fs.WIDTH))
    ax.fill_between(x, -r, r, color=col, alpha=0.12, lw=0)
    ax.plot(x, r, color=col, lw=1.1)
    ax.plot(x, -r, color=col, lw=1.1)
    ax.plot([tot, tot], [-R, R], color=col, lw=1.1)
    ax.axhline(0, color=fs.INK2, lw=0.5, ls=(0, (6, 2, 1, 2)))
    ax.axvline(xs, color=fs.INK2, lw=0.5, ls="--")
    ax.text(xs - 6, 0.55 * R, r"shoulder $x_S$", ha="right", va="center", fontsize=8)
    ax.annotate(r"tip sphere $r_n$", (0, 0), (25, -0.75 * R), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2))
    ax.text(tot / 2, 0.12 * R, "Haack profile", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-10, tot + 10)
    ax.set_ylim(-R - 10, R + 10)
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$r$ [mm]")
    fig.savefig(path)
    plt.close(fig)


def geometry_section(study: str, eq: dict, stem: str, sw: dict, tab: pd.DataFrame) -> str:
    mm = lambda v: f"{1e3 * v:.3f}"  # noqa: E731
    Ctxt = r"1/3" if eq["C"] else "0"
    n = math.ceil(len(tab) / 3)
    cols = [tab.iloc[i * n:(i + 1) * n].reset_index(drop=True) for i in range(3)]
    rows = []
    for i in range(n):
        cells = [f"{c.x_mm[i]:.3f} & {c.r_mm[i]:.3f}" if i < len(c) else " & " for c in cols]
        rows.append(" & ".join(cells) + r" \\")
    shape = NAMES[eq["shape"]]
    return rf"""
\section*{{Design {study}: {TITLES[study]}}}
{shape} profile, ${1e3 * (eq['L_phys']):.1f} + {1e3 * eq['c']:.0f} = {1e3 * eq['total_phys']:.1f}$\,mm. $x$ along
the axis from the blunted tip, $r$ the outer radius, mm.

\centerline{{\includegraphics[width=0.92\linewidth]{{figures/profile_{study}.pdf}}}}
\vspace{{-6pt}}
\begin{{align*}}
\text{{Spherical tip}}\quad & 0 \le x \le x_T: & r &= \sqrt{{r_n^2-(r_n-x)^2}}\\
\text{{Haack profile}}\quad & x_T \le x \le x_S: & r &= \frac{{R}}{{\sqrt{{\pi}}}}\sqrt{{\theta-\frac{{\sin 2\theta}}{{2}}+C\sin^3\theta}},\qquad
x=\frac{{L}}{{2}}\,(1-\cos\theta)-\delta,\quad \theta_T \le \theta \le \pi\\
\text{{Coupler cylinder}}\quad & x_S \le x \le x_S + c: & r &= R
\end{{align*}}

\begin{{minipage}}[t]{{0.44\linewidth}}\vspace{{0pt}}\small
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Parameter & Value\\
\midrule
Haack constant $C$ & ${Ctxt}$\\
Length to the sharp apex $L$ & {mm(eq['L'])}\\
Body radius $R$ & {mm(eq['R'])} ($D = {2e3 * eq['R']:.0f}$)\\
Tip radius $r_n$ & {mm(eq['rn'])}\\
Length removed by the tip $\delta$ & {mm(eq['delta'])}\\
Tangency point $(x_T, r_T)$ & ({mm(eq['x_t'])}, {mm(eq['r_t'])})\\
Tangency angle $\theta_T$ [rad] & {sw['theta_T']:.7f}\\
Shoulder $x_S$ & {mm(eq['L_phys'])}\\
Coupler cylinder $c$ & {mm(eq['c'])}\\
\textbf{{Total length}} & \textbf{{{mm(eq['total_phys'])}}}\\
\bottomrule
\end{{tabular}}

\smallskip
Tangent to the sphere at $x_T$ and to the cylinder at $x_S$ ($\mathrm{{d}}r/\mathrm{{d}}x = 0$).
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.53\linewidth}}\vspace{{0pt}}\scriptsize
\begin{{tabular}}{{@{{}}rr|rr|rr@{{}}}}
\toprule
$x$ & $r$ & $x$ & $r$ & $x$ & $r$\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{minipage}}

\subsection*{{SolidWorks}}
Sketch on the Front Plane, origin at the blunted tip, $x$ (rocket axis) to the right, document units mm.
\begin{{enumerate}}\setlength{{\itemsep}}{{1pt}}
\item \textbf{{Tip:}} Centerpoint Arc, centre $({1e3 * eq['rn']:.3f},\,0)$, from $(0,\,0)$ to $(x_T,\,r_T) = ({mm(eq['x_t'])},\,{mm(eq['r_t'])})$.
\item \textbf{{Haack profile:}} Tools $\rightarrow$ Sketch Entities $\rightarrow$ Equation Driven Curve $\rightarrow$
  \emph{{Parametric}}; paste (angles in radians, $t$ is $\theta$):
\begin{{Verbatim}}[frame=single,fontsize=\small,xleftmargin=0pt]
x_t = {sw['x']}
y_t = {sw['y']}
t1  = {sw['t1']}
t2  = {sw['t2']}
\end{{Verbatim}}
  The curve starts on the arc end $(x_T, r_T)$ and ends at $(x_S,\,R) = ({mm(eq['L_phys'])},\,{mm(eq['R'])})$;
  add a Tangent relation between the arc and the curve.
\item \textbf{{Coupler:}} Line from $({mm(eq['L_phys'])},\,{mm(eq['R'])})$ to $({mm(eq['total_phys'])},\,{mm(eq['R'])})$,
  then a line down to the axis and a centreline along the axis from the tip.
\item \textbf{{Solid:}} Revolved Boss, 360$^\circ$ about the centreline (outer mould line; shell and joint by ST).
\end{{enumerate}}
Without equation-driven curves: Insert $\rightarrow$ Curve $\rightarrow$ Curve Through XYZ Points with
the point file \texttt{{\_solidworks\_xyz.txt}} (tip to shoulder, $X$ along the axis, $Y = r$, $Z = 0$), then steps 3--4.\\
Files: \texttt{{{stem.replace('_', chr(92) + '_')}}} + \texttt{{.step}} (solid) and \texttt{{\_solidworks\_xyz.txt}} (points).
"""


def geometry_sheet(summary: dict, out: Path, date: str) -> tuple[Path, dict]:
    (HERE / "figures").mkdir(exist_ok=True)
    fs.setup()
    secs, checks = [], {}
    for study in ("A", "B"):
        eq = summary["equations"][study]
        rule = "inside" if study == "A" else "free"
        stem = summary["cad"][f"{rule}:{eq['shape']}"]["stem"]
        csv = CAD_DIR / f"{stem}.csv"
        sw = solidworks(eq)
        checks[study] = check_solidworks(eq, sw, csv)
        write_xyz(csv, CAD_DIR / f"{stem}_solidworks_xyz.txt")
        fig_profile(study, eq, csv, HERE / "figures" / f"profile_{study}.pdf")
        tab = pd.read_csv(out / f"radius_table_{study}.csv")
        tab = tab[~tab.x_mm.round(3).duplicated()].reset_index(drop=True)
        secs.append(geometry_section(study, eq, stem, sw, tab))
    tex = (PREAMBLE + r"\begin{document}" + "\n"
           + header("Weisshorn nose cone at 720 mm: geometry for SolidWorks", GEO_NAME, date)
           + "\n\\newpage\n".join(secs) + "\n\\end{document}\n")
    return compile_tex(GEO_NAME, tex), checks


# ============================================================================== loads and heating
CASES = [("A", "nominal"), ("A", "envelope"), ("B", "nominal"), ("B", "envelope")]
CASE_TEX = {"nominal": "nominal", "envelope": "hot day"}
ALPHAS = [1.0, 2.0, 5.0]   # incidences of the bending table [deg]
REQ10_C = 340.0 - 273.15   # Req_10: inside of the nose <= 340 K
BOND_C = 120.0             # DP490: shear strength drops above 120 degC (bond of the tip insert)


def ok(value: float, limit: float) -> str:
    return "ok" if value <= limit else r"\textbf{exceeds}"


def fig_force(out: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.32 * fs.WIDTH))
    for study, shape in (("A", "lvhaack"), ("B", "vonkarman")):
        d = pd.read_csv(out / f"joint_force_{study}.csv")
        for case, ls in (("nominal", "-"), ("envelope", (0, (4, 1.5)))):
            g = d[d.case == case]
            ax.plot(g.t, g.F, color=fs.COLORS[shape], ls=ls, label=f"{study}, {CASE_TEX[case]}")
    ax.axhline(tl.REQ06_N / tl.FOS, color=fs.INK2, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(19.6, tl.REQ06_N / tl.FOS + 12, r"Req\_06 / FoS", ha="right", fontsize=7, color=fs.INK2)
    ax.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
    ax.set(xlim=(0, 20), xlabel=r"time $t$ [s]", ylabel=r"axial force $F$ [N]")
    ax.legend(ncol=2, loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_temperatures(out: Path, path: Path) -> None:
    keys = [("T0", r"stagnation $T_0$ (air)", fs.INK2), ("T_shell_out_50mm", r"shell outer face, 50 mm", "#eb6834"),
            ("T_shell_in_50mm", r"shell inner face, 50 mm", "#2a78d6"), ("T_tip_insert", "Al tip insert", "#008300")]
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.34 * fs.WIDTH))
    for case, ls in (("nominal", "-"), ("envelope", (0, (4, 1.5)))):
        h = pd.read_csv(out / f"heating_history_A_{case}.csv")
        for k, lab, col in keys:
            ax.plot(h.t, h[k] - 273.15, color=col, ls=ls, label=lab if case == "nominal" else None)
    ax.axhline(REQ10_C, color=fs.INK2, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(43.5, REQ10_C + 3, r"Req\_10 (340 K)", ha="right", fontsize=7, color=fs.INK2)
    ax.set(xlim=(0, 44), ylim=(-50, 230), xlabel=r"time $t$ [s]", ylabel=r"temperature [$^\circ$C]")
    ax.legend(loc="upper right", ncol=2, title=r"design A; solid nominal, dashed hot day", title_fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def loads_sheet(summary: dict, out: Path, date: str) -> Path:
    (HERE / "figures").mkdir(exist_ok=True)
    fs.setup()
    fig_force(out, HERE / "figures" / "joint_force.pdf")
    fig_temperatures(out, HERE / "figures" / "temperatures_A.pdf")
    eqs, lds, hts = summary["equations"], summary["loads"], summary["heating"]
    FOS = tl.FOS
    A_ref = math.pi * eqs["A"]["R"] ** 2
    hist = {(s, c): pd.read_csv(out / f"heating_history_{s}_{c}.csv") for s, c in CASES}
    dp0 = {}
    for k, h in hist.items():
        p0, _ = stagnation_state(h.M.values, h.p.values, h["T"].values)
        dp0[k] = float(np.max(p0 - h.p.values))

    def lab(s, c):
        return f"{s} & {CASE_TEX[c]}"

    inp = []
    for s in ("A", "B"):
        e, ld = eqs[s], lds[s]
        inp.append(rf"{s} & {NAMES[e['shape']]} ${1e3 * e['L_phys']:.1f} + {1e3 * e['c']:.0f}$ & "
                   rf"{ld['m_payload']:.2f} & {ld['nominal']['m_nose']:.2f} & {1e3 * ld['nominal']['arm']:.0f} \\")
    ax_rows, bend_rows = [], []
    for s, c in CASES:
        ld = lds[s][c]
        ax_rows.append(rf"{lab(s, c)} & {ld['F_liftoff']:.0f} & {ld['F_max']:.0f} ({ld['t_F_max']:.1f}\,s) & "
                       rf"{FOS * ld['F_max']:.0f} & {ok(FOS * ld['F_max'], tl.REQ06_N)} & {ld['F_min']:.0f} & "
                       rf"{ld['req07']:.0f} & {ok(ld['req07'], 2000.0)} \\")
        ult = " & ".join(f"{FOS * a * ld['bend_per_deg']:.0f}" for a in ALPHAS)
        bend_rows.append(rf"{lab(s, c)} & {1e-3 * ld['q_max']:.1f} & {ld['M_q_max']:.2f} & {ld['bend_per_deg']:.1f} & "
                         rf"{ld['shear_per_deg']:.0f} & {ult} & {ld['alpha_req12']:.1f} \\")
    trow = []
    for key, name, lim in (("T0", r"Stagnation temperature $T_0$ (air)", None),
                           ("tip", r"Al 6082 tip insert (bond to the shell)", BOND_C),
                           ("shell_out_50", r"CFRP shell outer face, $x = 50$\,mm", None),
                           ("shell_50", r"CFRP shell inner face, $x = 50$\,mm", REQ10_C),
                           ("shell_mid", r"CFRP shell inner face, mid-length", REQ10_C),
                           ("shell_base", r"CFRP shell inner face, shoulder", REQ10_C)):
        vals = []
        for s, c in CASES:
            if key == "shell_out_50":
                v = float(hist[(s, c)]["T_shell_out_50mm"].max()) - 273.15
            else:
                v = hts[s][key][0 if c == "nominal" else 1]
            vals.append(v)
        lim_txt = "--" if lim is None else (f"{lim:.0f} (DP490)" if lim == BOND_C else f"{lim:.0f} (Req\\_10)")
        worst = "--" if lim is None else ok(max(vals), lim)
        trow.append(name + " & " + " & ".join(f"{v:.0f}" for v in vals) + rf" & {lim_txt} & {worst} \\")
    nl = "\n"
    a_lim = lds["A"]["envelope"]["alpha_req12"]
    p_txt = "; ".join(f"{s} {1e-3 * dp0[(s, 'nominal')]:.0f} / {1e-3 * dp0[(s, 'envelope')]:.0f} kPa"
                      for s in ("A", "B")) + " (nominal / hot day)"
    t_in = [hts["A"][k][0] for k in ("shell_50", "shell_mid", "shell_base")]
    body = rf"""
These are the loads at the nose joint and the temperatures of the 720\,mm nose (geometry sheet
2027\_C\_FD\_NOSE-CONE-720MM\_GEOMETRY), inputs for the ST design. They come from the FD Python
models of 2027\_C\_FD\_REPORT-2 (3-DoF ascent, semi-empirical drag, engineering heating). Two
trajectories: \emph{{nominal}} (ISA day, $I_{{sp}}$ masses) and \emph{{hot day}} (30\,$^\circ$C ground,
LVO masses, the envelope). \textbf{{Limit}} loads are the flight values; \textbf{{ultimate}} = limit
$\times$ FoS {FOS:g}, the ERT convention (e.g. Firehorn nose load case 2024\_C\_SE\_ST\_NOSECONE\_REQ\_10:
deceleration $\times$ dry mass $\times$ FoS 2). Requirements: Weisshorn sheet of 28.10.2026.

\section*{{1\quad Inputs}}
\begin{{center}}\small
\begin{{tabular}}{{@{{}}llrrr@{{}}}}
\toprule
Design & Profile [mm] & Payload [kg] & Nose mass $m_n$ [kg] & Arm $\ell$ CP--joint [mm]\\
\midrule
{nl.join(inp)}
\bottomrule
\end{{tabular}}
\end{{center}}
$m_n$ = nose with payload structure at the Req\_04 limit (1.40\,kg) + payload carried in the nose (A:
3U GRV 3.99\,kg + CanSat 0.35\,kg; B: CanSat). $A_\mathrm{{ref}} = \pi R^2 = {1e4 * A_ref:.1f}$\,cm$^2$.
$\ell$ = distance from the nose centre of pressure (slender body) to the joint plane.

\section*{{2\quad Axial force at the joint}}
Free body of the nose along the axis (compression $> 0$), with $T$ the thrust, $D$ the vehicle drag,
$m$ the vehicle mass and $D_n$ the nose drag:
\[
F(t) = m_n\,\frac{{T - D}}{{m}} + D_n ,\qquad
F_\mathrm{{rec}} = 20\,g\; m_n \times \mathrm{{FoS}} \quad\text{{(Req\_07, recovery deceleration, tension)}}.
\]
\begin{{center}}\small
\begin{{tabular}}{{@{{}}llrrrlrrl@{{}}}}
\toprule
& & \multicolumn{{4}}{{c}}{{Compression [N]}} & & \multicolumn{{2}}{{c}}{{Tension, recovery [N]}}\\
\cmidrule(lr){{3-6}}\cmidrule(l){{8-9}}
\multicolumn{{2}}{{@{{}}l}}{{Case}} & lift-off & max (limit) & ult. & Req\_06 300 & min $F$ & ult. & Req\_07 2000\\
\midrule
{nl.join(ax_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
\centerline{{\includegraphics[width=0.8\linewidth]{{figures/joint_force.pdf}}}}
The maximum is at burnout (highest acceleration and drag); the nose stays in compression during the
ascent ($F_\mathrm{{min}} \approx 0$), so the only tension case is the recovery load.

\section*{{3\quad Bending moment and shear at the joint}}
Normal force of the nose at incidence $\alpha$ (slender body, $C_{{N\alpha}} = 2$/rad) at the maximum
dynamic pressure, applied at the nose CP:
\[
V = C_{{N\alpha}}\,\alpha\, q_\mathrm{{max}}\, A_\mathrm{{ref}}, \qquad M_b = V\,\ell .
\]
\begin{{center}}\small
\begin{{tabular}}{{@{{}}llrrrrrrrr@{{}}}}
\toprule
& & & & \multicolumn{{2}}{{c}}{{Limit, per degree}} & \multicolumn{{3}}{{c}}{{$M_b$ ultimate [N\,m]}} & \\
\cmidrule(lr){{5-6}}\cmidrule(lr){{7-9}}
\multicolumn{{2}}{{@{{}}l}}{{Case}} & $q_\mathrm{{max}}$ [kPa] & $M$ & $M_b$ [N\,m/$^\circ$] & $V$ [N/$^\circ$] &
$\alpha = 1^\circ$ & $2^\circ$ & $5^\circ$ & $\alpha$ at Req\_12 [$^\circ$]\\
\midrule
{nl.join(bend_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
Req\_12 (200\,N\,m ultimate) is reached at $\alpha = {a_lim:.1f}^\circ$ for design A on the hot day:
the incidence that ST designs for (gusts, weathercocking) decides whether Req\_12 holds.

\section*{{4\quad Wall pressure}}
The highest wall pressure is at the stagnation point of the tip, $p_0 - p_\infty$ behind the normal
shock (max.\ at burnout), an upper bound for the whole shell: {p_txt}. The pressure distribution
on the shell, $C_p(x)$, follows from the Fluent runs of REPORT-5.

\section*{{5\quad Temperatures}}
Laminar stagnation-point heating (Fay--Riddell) on the $r_n = 4.5$\,mm tip, turbulent
(Reynolds analogy, Eckert reference temperature) downstream; the Al 6082 tip insert (80\,mm, 60\,g)
as a lumped mass, the 1.2\,mm CFRP shell as 1D conduction with an adiabatic inner face (conservative,
bare shell without the insert). Peak values over the flight [$^\circ$C]:
\begin{{center}}\small
\begin{{tabular}}{{@{{}}lrrrrll@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{A}} & \multicolumn{{2}}{{c}}{{B}} & & \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Location & nominal & hot day & nominal & hot day & Limit & Worst case\\
\midrule
{nl.join(trow)}
\bottomrule
\end{{tabular}}
\end{{center}}
\centerline{{\includegraphics[width=0.8\linewidth]{{figures/temperatures_A.pdf}}}}
Materials are the REPORT-1 assumptions (Al 6082 insert, CFRP shell 1.2\,mm, DP490 bond): to be
confirmed by ST. Fluent heat-transfer runs and an Ansys transient-thermal model of the tip will
replace these values in 2027\_C\_FD\_REPORT-5 (12.11.2026).

\section*{{6\quad Open points for ST and SE}}
\begin{{itemize}}\setlength{{\itemsep}}{{1pt}}
\item \textbf{{Req\_06:}} the ultimate compression exceeds 300\,N for both designs
  (A {FOS * lds['A']['envelope']['F_max']:.0f}\,N, B {FOS * lds['B']['envelope']['F_max']:.0f}\,N, hot day).
  \textbf{{Req\_07:}} with the 3U GRV in the nose, the recovery tension of A
  ({lds['A']['nominal']['req07']:.0f}\,N) exceeds 2000\,N.
\item \textbf{{Req\_12:}} the design incidence at max $q$ must be agreed (Req\_12 is reached at
  $\approx {a_lim:.0f}^\circ$).
\item \textbf{{Req\_10:}} does 340\,K apply to the air in the nose or to the inner wall? The bare inner
  wall of A reaches {min(t_in):.0f}--{max(t_in):.0f}\,$^\circ$C (nominal) against 67\,$^\circ$C.
\item \textbf{{Req\_08:}} the TBD values (max temperature, duration) can be filled from section 5.
\end{{itemize}}
"""
    tex = (PREAMBLE + r"\begin{document}" + "\n"
           + header("Weisshorn nose cone at 720 mm: joint loads and temperatures", LOADS_NAME, date)
           + body + "\\end{document}\n")
    return compile_tex(LOADS_NAME, tex)


# ============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--only", choices=["geometry", "loads"])
    ap.add_argument("--date", default="30.10.2026")
    a = ap.parse_args()
    mdl = tl.Model(Path(a.config))
    out = Path(mdl.cfg.project.output_dir) / "final"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if a.only in (None, "geometry"):
        pdf, checks = geometry_sheet(summary, out, a.date)
        print(pdf)
        print(json.dumps(checks, indent=1))
    if a.only in (None, "loads"):
        print(loads_sheet(summary, out, a.date))


if __name__ == "__main__":
    main()

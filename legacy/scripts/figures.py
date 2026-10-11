"""All report figures. usage: python figures.py [name ...]  (default: all)"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import COLORS, STYLES, MARKERS, INK, INK2, GRID  # noqa: E402
from geometry import SHAPES, surface, radius, blunt_tangent_point  # noqa: E402
from taylor_maccoll import cone_cp  # noqa: E402
import analysis as an  # noqa: E402
from flight import simulate, LVO  # noqa: E402
import thermal as th  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
R = an.R

# final design (set from the trade study, see report)
FINAL = dict(shape="vonkarman", f=4.5, rn_over_R=0.05)

SHORT = {"cone": "Cone", "ogive": "Tangent ogive", "parabolic": "Parabolic", "power050": r"Power $n=\tfrac12$",
         "power075": r"Power $n=\tfrac34$", "lvhaack": "LV-Haack", "vonkarman": r"Von K\'arm\'an"}


def shape_legend(fig, ncol=4, y=1.0, shapes=SHAPES, marker=False):
    h = [Line2D([], [], color=COLORS[s], ls=STYLES[s], marker=MARKERS[s] if marker else None,
                label=SHORT[s]) for s in shapes]
    fig.legend(handles=h, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.4)


def sweep(study="main"):
    df = pd.read_csv(os.path.join(RES, "euler_results.csv"))
    return df[df.study == study]


# ------------------------------------------------------------------ 1 profiles
def fig_profiles():
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.36 * fs.WIDTH),
                               gridspec_kw=dict(width_ratios=[2.3, 1]))
    L = 2 * R * 4.0
    for s in SHAPES:
        x, y = surface(s, L, R, n=800)
        a.plot(1e3 * x, 1e3 * y, color=COLORS[s], ls=STYLES[s])
    a.axhline(1e3 * an.R_3U, color=INK2, lw=0.6, ls=(0, (1, 2)))
    a.text(8, 1e3 * an.R_3U + 2.5, r"3U section needs $r \geq 77$ mm", color=INK2, fontsize=7.5)
    a.set(xlim=(0, 1e3 * L), ylim=(0, 100), xlabel=r"$x$ from tip [mm]", ylabel=r"$r$ [mm]")
    fs.panel(a, "a", y=0.97, x=0.9)
    # tip zoom with a 9 mm spherical blunting on the Von Karman profile
    for s in SHAPES:
        x, y = surface(s, L, R, n=3000)
        b.plot(1e3 * x, 1e3 * y, color=COLORS[s], ls=STYLES[s], lw=1.0)
    rn = FINAL["rn_over_R"] * R
    x, y = surface("vonkarman", L, R, rn=rn, n=3000)
    b.fill_between(1e3 * x, 0, 1e3 * y, color=COLORS["vonkarman"], alpha=0.12, lw=0)
    b.plot(1e3 * x, 1e3 * y, color=INK, lw=0.8)
    xt, xc = blunt_tangent_point("vonkarman", L, R, rn)
    b.annotate(r"$r_n = 4.5$ mm", xy=(1e3 * (xc - rn * 0.75), 1e3 * rn * 0.66), xytext=(24, 31),
               fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    b.set(xlim=(0, 60), ylim=(0, 38), xlabel=r"$x$ [mm]")
    b.set_aspect("equal")
    fs.panel(b, "b", y=0.97, x=0.03)
    shape_legend(fig, ncol=4, y=0.98)
    fs.save(fig, "fig_profiles", FIG)


# ---------------------------------------------------------------- 2 validation
def fig_validation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    d = sweep("cones")
    dm = sweep("main")
    dm = dm[(dm["shape"] == "cone")]
    angles = {}
    for f in sorted(d.fineness.unique()):
        angles[np.degrees(np.arctan(1 / (2 * f)))] = d[d.fineness == f]
    angles[np.degrees(np.arctan(1 / 8))] = dm[dm.fineness == 4.0]
    seq = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    for (thc, dd), c in zip(sorted(angles.items()), seq):
        Ms = np.linspace(1.12, 2.05, 60)
        cp = []
        for M in Ms:
            try:
                cp.append(cone_cp(M, thc)[0])
            except Exception:
                cp.append(np.nan)
        a.plot(Ms, cp, color=c, lw=1.2)
        a.plot(dd.mach, dd.cd, "o", mfc="white", mec=c, mew=1.0, ms=4.5, ls="none")
        i = np.nanargmax(Ms) if True else 0
        a.text(2.07, cp[-1], rf"${thc:.1f}^\circ$", va="center", fontsize=7.5, color=INK)
    a.set(xlim=(1.1, 2.25), xlabel=r"$M_\infty$", ylabel=r"cone surface $C_p$ $(= C_{D,\mathrm{w}})$")
    a.legend(handles=[Line2D([], [], color=INK2, label="Taylor--Maccoll (exact)"),
                      Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="Euler (this work)")],
             loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=7)
    fs.panel(a, "a", x=0.90, y=0.96)
    # grid convergence with Richardson extrapolation (Von Karman, L/D = 4)
    g = pd.concat([sweep("grid"), dm0 := sweep("main")])
    g = g[(g["shape"] == "vonkarman") & (g.fineness == 4.0) & (g.rn == 0)]
    ncell = {"coarse": 80 * 40, "medium": 160 * 80, "fine": 320 * 160}
    for M, ls, c in [(1.4, "-", "#4a3aa7"), (1.2, "--", "#2a78d6")]:
        gg = g[(g.mach == M) & g.grid.isin(list(ncell))].drop_duplicates("grid")
        if len(gg) < 2:
            continue
        hh = np.array([1 / np.sqrt(ncell[k]) for k in gg.grid])
        o = np.argsort(hh)
        hh, cc = hh[o], gg.cd.values[o]
        if len(cc) == 3:   # observed order, Roache GCI with Fs = 1.25
            p = np.log(abs((cc[2] - cc[1]) / (cc[1] - cc[0]))) / np.log(2)
            f0, f1, Fs, im = cc[0], cc[1], 1.25, 1
        else:              # two grids: formal order 2, Fs = 3
            p, f0, f1, Fs, im = 2.0, cc[0], cc[1], 3.0, 0
        cex = f0 + (f0 - f1) / (2**p - 1)
        # GCI of the production (medium) grid
        gci = (Fs * abs((f1 - f0) / f1) * 2**p / (2**p - 1) if len(cc) == 3
               else Fs * abs((f1 - f0) / f0) / (2**p - 1)) * 100
        tag = rf"$p={p:.1f}$" if len(cc) == 3 else r"$p=2$ (formal)"
        lab = rf"$M_\infty={M}$: {tag}, GCI$={gci:.1f}\,\%$"
        b.plot(np.concatenate([[0], hh]), np.concatenate([[cex], cc]) / cex, color=c, ls=ls,
               marker="o", ms=4.5, mfc="white", label=lab)
        b.plot(hh[im], cc[im] / cex, "o", color=c, ms=4.5)
    b.axhline(1, color=GRID, lw=0.6, zorder=0)
    b.set(xlim=(-0.001, None), xlabel=r"grid spacing $h = N_{\rm cells}^{-1/2}$",
          ylabel=r"$C_{D,\mathrm{w}} / C_{D,\mathrm{w}}^{\,h\to 0}$")
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=6.8, handlelength=2.2)
    fs.panel(b, "b", x=0.015)
    fig.tight_layout(w_pad=1.5)
    fs.save(fig, "fig_validation", FIG)


# ------------------------------------------------------ 3 Cd_w vs Mach and f
def fig_cdw():
    df = sweep("main")
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH))
    Mf = np.linspace(0.6, 1.65, 300)
    a.axvspan(an.M_ONSET, 1.1, color=GRID, alpha=0.45, lw=0)
    a.text(0.95, 0.121, "transonic:\nmodelled", ha="center", va="top", fontsize=7, color=INK2)
    for s in SHAPES:
        try:
            w = an.WaveDrag(df, s, 4.0)
        except ValueError:
            continue
        a.plot(Mf, w(Mf), color=COLORS[s], ls=STYLES[s])
        d = df[(df["shape"] == s) & (df.fineness == 4.0)]
        a.plot(d.mach, d.cd, ls="none", marker=MARKERS[s], color=COLORS[s], mfc="white", ms=4)
    a.set(xlim=(0.6, 1.65), ylim=(0, 0.125), xlabel=r"$M_\infty$", ylabel=r"nose wave drag $C_{D,\mathrm{w}}$ ($L/D=4$)")
    fs.panel(a, "a", x=0.03)
    ff = np.array([3, 4, 5.0])
    for s in SHAPES:
        d = df[(df["shape"] == s) & (df.mach == 1.4) & (df.fineness <= 5)].sort_values("fineness")
        b.plot(d.fineness, d.cd, color=COLORS[s], ls=STYLES[s], marker=MARKERS[s], mfc="white", ms=4)
    b.plot(ff, 1 / ff**2, color=INK, lw=0.8, ls=(0, (1, 1)))
    b.annotate("slender-body theory,\n" + r"Von K\'arm\'an: $1/(L/D)^2$", xy=(3.5, 1 / 3.5**2),
               xytext=(3.04, 0.0345), fontsize=7, color=INK, ha="left", va="bottom",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    b.set(xscale="log", yscale="log", ylim=(0.03, 0.14), xlabel=r"fineness $L/D$",
          ylabel=r"$C_{D,\mathrm{w}}$ at $M_\infty=1.4$")
    b.set_xticks([3, 4, 5])
    b.set_xticklabels(["3", "4", "5"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_yticks([0.04, 0.06, 0.08, 0.1, 0.12])
    b.set_yticklabels(["0.04", "0.06", "0.08", "0.10", "0.12"])
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fs.panel(b, "b", x=0.90, y=0.96)
    shape_legend(fig, ncol=4, y=0.98, marker=True)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.98))
    fs.save(fig, "fig_cdw", FIG)


# ----------------------------------------------------------- 4 wall pressure
def fig_cp():
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH), sharey=True)
    for ax, M, let in [(a, 1.1, "a"), (b, 1.4, "b")]:
        for s in SHAPES:
            fn = os.path.join(RES, "cp", f"main_{s}_f4_rn0_medium_M{M:g}.npz")
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            x, cp, ish = z["x"], z["cp"], int(z["ish"])
            Lx = x[ish - 1] + 0.5 * (x[ish] - x[ish - 1])
            sel = x < Lx + 2.5 * 2
            ax.plot(x[sel] / 8.0, cp[sel], color=COLORS[s], ls=STYLES[s])
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(1, color=INK2, lw=0.5, ls=(0, (1, 2)))
        ax.text(1.02, 0.42, "shoulder", fontsize=7, color=INK2, rotation=90, va="top")
        ax.set(xlim=(0, 1.35), ylim=(-0.25, 0.45), xlabel=r"$x/L$")
        fs.panel(ax, let, x=0.03, y=0.09)
        ax.text(0.12, 0.055, rf"$M_\infty = {M}$", transform=ax.transAxes, fontsize=8)
    a.set_ylabel(r"wall pressure coefficient $C_p$")
    shape_legend(fig, ncol=4, y=0.98)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.98))
    fs.save(fig, "fig_cp", FIG)


# ---------------------------------------------------------------- 5 trajectory
def trajectories(k):
    df = sweep("main")
    veh = an.make_vehicle(df, *an.BASELINE, k)
    nom = simulate(veh, mass_model="isp", dt_out=0.02)
    env = simulate(veh, mass_model="lvo", T_ground=303.15, dt_out=0.02)
    return veh, nom, env


def fig_trajectory(k):
    veh, nom, env = trajectories(k)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    for tr, ls, lab, dy in [(nom, "-", r"nominal (ISA, $m_{\rm bo}=54$ kg)", -0.32),
                            (env, "--", r"envelope ($30^\circ$C, $m_{\rm bo}=45$ kg)", -0.08)]:
        a.plot(tr["t"], tr["M"], color="#2a78d6", ls=ls, label=lab)
        i = np.argmax(tr["M"])
        a.plot(tr["t"][i], tr["M"][i], "o", color="#2a78d6", ms=4)
        a.annotate(rf"$M={tr['M'][i]:.2f}$ at $z={tr['z'][i]/1e3:.1f}$ km", xy=(tr["t"][i], tr["M"][i]),
                   xytext=(tr["t"][i] + (2.5 if dy < -0.2 else 5.0), tr["M"][i] + dy), fontsize=7,
                   arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    a.axhspan(an.M_ONSET, 1.1, color=GRID, alpha=0.45, lw=0)
    a.set(xlim=(0, 40), ylim=(0, 2.1), xlabel=r"$t$ [s]", ylabel=r"$M$")
    a.legend(loc="upper right", fontsize=7)
    fs.panel(a, "a", x=0.93, y=0.08)
    # where is the drag impulse spent? (nominal flight, baseline nose)
    bins = np.arange(0, 1.75, 0.1)
    tot, nose = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    t, M = nom["t"], nom["M"]
    dt = np.gradient(t)
    for kk in range(len(t)):
        Re = nom["rho"][kk] * nom["V"][kk] / nom["mu"][kk]
        cdt = veh.cd(M[kk], Re, t[kk] < LVO["t_burn"])
        cdn = float(veh.cdw(M[kk]))
        j = np.searchsorted(bins, M[kk]) - 1
        if 0 <= j < len(tot):
            tot[j] += nom["q"][kk] * veh.Aref * cdt * dt[kk]
            nose[j] += nom["q"][kk] * veh.Aref * cdn * dt[kk]
    T = tot.sum()
    b.bar(bins[:-1], tot / T * 100, width=0.1, align="edge", color="#b7d3f6", edgecolor="white", lw=1,
          label="whole vehicle")
    b.bar(bins[:-1], nose / T * 100, width=0.1, align="edge", color="#4a3aa7", edgecolor="white", lw=1,
          label=r"nose wave drag (Von K\'arm\'an, $L/D=4$)")
    b.set(xlim=(0, 1.7), ylim=(0, 1.38 * (tot / T * 100).max()), xlabel=r"$M$",
          ylabel=r"share of drag impulse $\int D\,\mathrm{d}t$ [\%]")
    b.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 1))
    b.text(0.03, 0.62, rf"nose wave drag: {nose.sum()/T*100:.0f}\,\% of $\int D\,\mathrm{{d}}t$",
           transform=b.transAxes, ha="left", fontsize=7.5)
    fs.panel(b, "b", x=0.92, y=0.08)
    fig.tight_layout(w_pad=1.5)
    fs.save(fig, "fig_trajectory", FIG)
    return nom, env


# -------------------------------------------------------------------- 6 trade
def fig_trade(t):
    fig, axs = plt.subplots(1, 3, figsize=(fs.WIDTH, 0.36 * fs.WIDTH))
    for s in SHAPES:
        d = t[t["shape"] == s].sort_values("fineness")
        kw = dict(color=COLORS[s], ls=STYLES[s], marker=MARKERS[s], mfc="white", ms=3.5)
        axs[0].plot(d.fineness, d.d_apogee, **kw)
        axs[1].plot(d.fineness, d.l3U_mm, **kw)
        axs[2].plot(d.fineness, d.dSM_cal, **kw)
    axs[0].set(ylabel=r"$\Delta z_{\rm apogee}$ vs.\ baseline [m]")
    axs[1].axhline(1e3 * an.L_3U, color=INK2, lw=0.6, ls=(0, (1, 2)))
    axs[1].text(3.05, 1e3 * an.L_3U + 6, "full 3U length", fontsize=7, color=INK2)
    axs[1].set(ylabel=r"length with $r\geq 77$ mm [mm]", ylim=(0, 520))
    axs[2].axhline(0, color=GRID, lw=0.6, zorder=0)
    axs[2].set(ylabel=r"$\Delta$ static margin [cal]")
    for ax, let in zip(axs, "abc"):
        ax.set_xlabel(r"fineness $L/D$")
        ax.plot(an.BASELINE[1], 0 if ax is not axs[1] else np.nan, "x", color=INK, ms=5)
        fs.panel(ax, let, x=0.04)
    shape_legend(fig, ncol=4, y=0.98, marker=True)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.98))
    fs.save(fig, "fig_trade", FIG)


# ----------------------------------------------------------------- 7 bluntness
def fig_bluntness(nom, env):
    d = pd.concat([sweep("blunt"), sweep("main")])
    d = d[(d["shape"] == "vonkarman") & (d.fineness == 4.0) & (d.grid == "medium")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    for M, c, ls in [(1.2, "#9ec5f4", "--"), (1.4, "#3987e5", "-"), (1.6, "#104281", "-.")]:
        dd = d[d.mach == M].sort_values("rn").drop_duplicates("rn")
        if dd.empty:
            continue
        a.plot(dd.rn, dd.cd / dd.cd.values[0], color=c, ls=ls, marker="o", mfc="white", ms=4,
               label=rf"$M_\infty = {M}$")
    a.axhline(1, color=GRID, lw=0.6, zorder=0)
    a.axvline(FINAL["rn_over_R"], color=INK2, lw=0.5, ls=(0, (1, 2)))
    a.set(xlabel=r"bluntness $r_n/R$", ylabel=r"$C_{D,\mathrm{w}}(r_n) / C_{D,\mathrm{w}}(r_n=0)$")
    a.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.05, 0.93))
    a.text(0.97, 0.06, r"Von K\'arm\'an, $L/D=4$", transform=a.transAxes, ha="right", fontsize=7.5)
    fs.panel(a, "a", x=0.03)
    # stagnation-point heating vs nose radius at the envelope max-Mach point
    rn = np.linspace(2e-3, 20e-3, 60)
    for tr, ls, lab in [(nom, "-", "nominal"), (env, "--", "envelope")]:
        i = np.argmax(tr["M"])
        h = np.array([th.h_stagnation(tr["M"][i], tr["p"][i], tr["T"][i], r)[0] for r in rn])
        b.plot(1e3 * rn, h / 1e3, color="#2a78d6", ls=ls, label=lab)
    b.plot(1e3 * rn, 1.6 * (9e-3 / rn) ** 0.5, color=INK, lw=0.6, ls=(0, (1, 1.5)))
    b.text(14, 1.6 * (9 / 14) ** 0.5 * 1.08, r"$\propto r_n^{-1/2}$", fontsize=7.5)
    b.axvline(1e3 * FINAL["rn_over_R"] * R, color=INK2, lw=0.5, ls=(0, (1, 2)))
    b.set(xlabel=r"tip radius $r_n$ [mm]", ylabel=r"$h_s$ at $M_{\max}$ [kW/(m$^2$K)]")
    b.legend(loc="upper right", fontsize=7)
    fs.panel(b, "b", x=0.03)
    fig.tight_layout(w_pad=1.5)
    fs.save(fig, "fig_bluntness", FIG)


# -------------------------------------------------------------------- 8 thermal
def fig_thermal(nom, env, L):
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    stations = [(0.05, "#2a78d6", r"$x=50$ mm"), (0.5 * L, "#eb6834", r"$x=L/2$"), (L, "#1baf7a", r"$x=L$ (shoulder)")]
    for tr, ls in [(nom, "-"), (env, "--")]:
        for x, c, lab in stations:
            r = th.wall_response(tr, "CFRP (through-thickness)", 0.0012, x)
            a.plot(r["t"], r["T_in"] - 273.15, color=c, ls=ls)
        p0, T0 = th.stagnation_state(tr["M"], tr["p"], tr["T"])
        a.plot(tr["t"], T0 - 273.15, color=INK, lw=0.7, ls=ls)
    a.axhline(120, color="#d03b3b", lw=0.7, ls=(0, (4, 2)))
    a.text(29.5, 123, r"DP490 bond limit", fontsize=7, color=INK, ha="right")
    a.set(xlim=(0, 30), ylim=(0, 175), xlabel=r"$t$ [s]", ylabel=r"temperature [$^\circ$C]")
    a.legend(handles=[Line2D([], [], color=c, label=lab) for _, c, lab in stations]
             + [Line2D([], [], color=INK2, ls="-", label="nominal"), Line2D([], [], color=INK2, ls="--", label="envelope")],
             loc="lower left", fontsize=6.8, ncol=3, bbox_to_anchor=(0, 1.01), columnspacing=1.0)
    fs.panel(a, "a", x=0.03)
    rn = FINAL["rn_over_R"] * R
    for tr, ls in [(nom, "-"), (env, "--")]:
        r1 = th.wall_response(tr, "PET-CF (FDM)", 0.004, "stagnation", rn=rn)
        r2 = th.tip_insert(tr, FINAL["shape"], 2 * R * FINAL["f"], R, rn, 0.08, an.M_TIP_AL)
        b.plot(r1["t"], r1["T_out"] - 273.15, color="#eb6834", ls=ls)
        b.plot(r2["t"], r2["T"] - 273.15, color="#2a78d6", ls=ls)
        p0, T0 = th.stagnation_state(tr["M"], tr["p"], tr["T"])
        b.plot(tr["t"], T0 - 273.15, color=INK, lw=0.7, ls=ls)
    b.axhspan(70, 80, color="#eb6834", alpha=0.15, lw=0)
    b.text(29.5, 83, "PET-CF HDT range", fontsize=7, ha="right", color=INK)
    b.legend(handles=[Line2D([], [], color="#eb6834", label="PET-CF printed tip (surface)"),
                      Line2D([], [], color="#2a78d6", label="Al 6082 machined tip (60 g)")],
             loc="lower left", fontsize=6.8, bbox_to_anchor=(0, 1.01), ncol=1)
    b.set(xlim=(0, 30), xlabel=r"$t$ [s]")
    fs.panel(b, "b", x=0.03)
    fig.tight_layout(w_pad=1.0)
    fs.save(fig, "fig_thermal", FIG)


# ---------------------------------------------------------------- 9 final design
def fig_final():
    s, f, rnR = FINAL["shape"], FINAL["f"], FINAL["rn_over_R"]
    L = 2 * R * f
    rn = rnR * R
    x, y = surface(s, L, R, rn=rn, n=1500)
    xt, xc = blunt_tangent_point(s, L, R, rn)
    x0 = x[0]
    fig, a = plt.subplots(figsize=(fs.WIDTH, 0.30 * fs.WIDTH))
    X, Y = 1e3 * (x - x0), 1e3 * y
    sh = 60.0  # shoulder (bonded coupler) length
    for sgn in (1, -1):
        a.plot(X, sgn * Y, color=INK, lw=1.0)
        a.plot([X[-1], X[-1] + sh], [sgn * 90, sgn * 90], color=INK, lw=1.0)
    a.plot([X[-1] + sh] * 2, [-90, 90], color=INK, lw=1.0)
    a.plot([X[-1]] * 2, [-90, 90], color=INK2, lw=0.5, ls=(0, (3, 2)))
    # Al tip insert up to x_tip
    x_tip = 80.0
    sel = X <= x_tip
    a.fill_between(X[sel], -Y[sel], Y[sel], color="#86b6ef", lw=0)
    a.annotate("Al 6082 tip insert", xy=(x_tip * 0.6, -8), xytext=(x_tip * 0.6, -80), fontsize=6.5, ha="center",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    # 3U payload envelope
    l3 = 1e3 * an.L_3U
    x3 = X[-1] + sh - l3 - 5
    a.add_patch(plt.Rectangle((x3, -50), l3, 100, fill=False, ec=COLORS["ogive"], lw=0.8, ls=(0, (4, 2))))
    a.text(x3 + l3 / 2, 0, "3U CubeSat envelope", fontsize=6.5, ha="center", va="center", color=INK)
    a.annotate("", xy=(0, 105), xytext=(X[-1], 105), arrowprops=dict(arrowstyle="<->", lw=0.6))
    a.text(X[-1] / 2, 110, rf"{X[-1]:.0f} mm tip to shoulder (profile length $L = {1e3 * L:.0f}$ mm)",
           ha="center", fontsize=7.5)
    a.annotate("", xy=(X[-1] + sh + 15, -90), xytext=(X[-1] + sh + 15, 90), arrowprops=dict(arrowstyle="<->", lw=0.6))
    a.text(X[-1] + sh + 22, 0, r"$\varnothing 180$", rotation=90, va="center", fontsize=7.5)
    a.text(X[-1] + sh / 2, -100, rf"shoulder {sh:.0f} mm", ha="center", va="top", fontsize=6.5)
    a.set_aspect("equal")
    a.axis("off")
    a.set(xlim=(-10, X[-1] + sh + 40), ylim=(-112, 122))
    fs.save(fig, "fig_final", FIG)
    return L, x0


def main():
    fs.setup()
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["profiles", "validation", "cdw", "cp", "trajectory", "trade", "bluntness",
                             "thermal", "final"]
    k = float(open(os.path.join(RES, "calibration.txt")).read().split("=")[1])
    nom = env = None
    for w in which:
        print("figure", w, flush=True)
        if w in ("trajectory", "bluntness", "thermal") and nom is None:
            _, nom, env = trajectories(k)
        if w == "profiles":
            fig_profiles()
        elif w == "validation":
            fig_validation()
        elif w == "cdw":
            fig_cdw()
        elif w == "cp":
            fig_cp()
        elif w == "trajectory":
            fig_trajectory(k)
        elif w == "trade":
            fig_trade(pd.read_csv(os.path.join(RES, "trade_table.csv")))
        elif w == "bluntness":
            fig_bluntness(nom, env)
        elif w == "thermal":
            fig_thermal(nom, env, 2 * R * FINAL["f"])
        elif w == "final":
            fig_final()


if __name__ == "__main__":
    main()

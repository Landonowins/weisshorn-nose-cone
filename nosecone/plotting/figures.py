"""All figures of the tool. Each function draws one figure from a :class:`PlotContext`.

Cosmetic settings (label positions, fixed axis ranges of the non-dimensional plots, colours)
are collected in :data:`PLOT_DEFAULTS`; any of them can be overridden per plot in the
configuration (``output.plot_options``). Axis ranges that depend on the vehicle (flight time,
temperatures, payload length) are automatic unless given there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .. import heating as ht  # noqa: E402
from ..config import Config, parse_station  # noqa: E402
from ..design import Study, mach_key  # noqa: E402
from ..euler import GridSpec, make_grid  # noqa: E402
from ..flight import Atmosphere, burnout_mass  # noqa: E402
from ..geometry import Profile, base_angle_deg, blunt_tangent_point, radius, slope, surface  # noqa: E402
from ..taylor_maccoll import cone_cp  # noqa: E402
from . import style as fs  # noqa: E402
from .style import GRID, INK, INK2, tex  # noqa: E402

# ------------------------------------------------------------------------- plot defaults
PLOT_DEFAULTS: dict[str, dict[str, Any]] = {
    "profiles": dict(fineness=None, label_x_mm=8.0, label_dy_mm=2.5, ylim_mm=100.0, tip_xlim_mm=60.0,
                     tip_ylim_mm=38.0, tip_label_mm=(24.0, 31.0)),
    "validation": dict(mach_range=(1.12, 2.05), xlim=(1.1, 2.25),
                       colors=("#9ec5f4", "#5598e7", "#256abf", "#0d366b"),
                       grid_styles=(("-", "#4a3aa7"), ("--", "#2a78d6"))),
    "cdw": dict(fineness=None, mach_range=(0.6, 1.65), ylim=(0, 0.125), fineness_ylim=(0.03, 0.14),
                fineness_mach=1.4, yticks=(0.04, 0.06, 0.08, 0.1, 0.12)),
    "cp": dict(fineness=None, machs=(1.1, 1.4), xlim=(0, 1.35), ylim=(-0.25, 0.45), behind_shoulder=5.0),
    "trajectory": dict(xlim=None, ylim=None, mach_bins_max=None, bin_width=0.1, color="#2a78d6"),
    "breakdown": dict(altitude=3000.0, mach_range=None, annotate_mach=1.4),
    "trade": dict(payload_ylim=None),
    "robustness": dict(markers=("o", "v", "^", "s", "D", "P", "X"),
                       colors=("#0b0b0b", "#9ec5f4", "#256abf", "#86b6ef", "#1c5cab", "#eb6834", "#d03b3b")),
    "bluntness": dict(machs=(1.2, 1.4, 1.6), colors=("#9ec5f4", "#3987e5", "#104281"), styles=("--", "-", "-."),
                      guide=None, guide_label_r=None),
    "thermal": dict(xlim=None, ylim=None, colors=("#2a78d6", "#eb6834", "#1baf7a", "#7f3f98", "#b8860b")),
    "final": dict(),
    "gallery": dict(mach=1.4),
    "fields": dict(machs=(1.4,), xlim=None, ylim=(-3.2, 3.2), clim=(-0.25, 0.0, 0.45)),
    "grid": dict(level="coarse", ylim=(-0.3, 22), zoom_xlim=(-0.4, 2.0), zoom_ylim=(-0.02, 1.0)),
    "schematics": dict(fineness=2.0),
    "renders": dict(),
}
"""Cosmetic defaults of every plot (overridable in output.plot_options.<plot>)."""

ALL_PLOTS = ["profiles", "schematics", "validation", "cdw", "cp", "trajectory", "breakdown", "trade",
             "robustness", "bluntness", "thermal", "final", "gallery", "fields", "grid", "renders"]


@dataclass
class PlotContext:
    """Data shared by the plots."""

    cfg: Config
    study: Study
    trade: Optional[pd.DataFrame] = None
    robustness: Optional[pd.DataFrame] = None
    traj: Optional[tuple] = None
    outdir: Path = Path("figures")
    log: Callable[[str], None] = print
    saved: list = field(default_factory=list)

    def opt(self, plot: str, key: str) -> Any:
        """Plot option: output.plot_options.<plot>.<key>, else PLOT_DEFAULTS."""
        user = self.cfg.output.plot_options.get(plot, {}) or {}
        return user[key] if key in user else PLOT_DEFAULTS.get(plot, {}).get(key)

    @property
    def R(self) -> float:
        """Base radius [m]."""
        return self.cfg.R

    @property
    def s(self) -> float:
        """Drawing scale of the dimensional plots relative to the 180 mm Weisshorn (positions in mm)."""
        return 1e3 * self.R / 90.0

    def p(self, name: str) -> Profile:
        """Profile object of a name."""
        return self.study.profiles[name]

    def order(self) -> list[str]:
        """Profiles of the sweep, in configuration order."""
        return list(self.cfg.nose.profiles)

    def sty(self, name: str) -> tuple:
        """(colour, line style, marker) of a profile."""
        return fs.style_of(name, self.order())

    def label(self, name: str) -> str:
        """Plot label of a profile."""
        return tex(self.p(name).label)

    def save(self, fig: Any, name: str) -> None:
        """Save a figure in every format of output.formats."""
        o = self.cfg.output
        self.saved += fs.save(fig, name, self.outdir, o.formats, o.png_dpi)


def shape_legend(ctx: PlotContext, fig: Any, ncol: int = 4, y: float = 1.0, marker: bool = False,
                 shapes: Optional[list] = None) -> None:
    """Figure legend with one entry (colour, style, marker) per profile."""
    h = []
    for s in shapes or ctx.order():
        c, ls, mk = ctx.sty(s)
        h.append(Line2D([], [], color=c, ls=ls, marker=mk if marker else None, label=ctx.label(s)))
    fig.legend(handles=h, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.4)


def _db(ctx: PlotContext, study: str) -> pd.DataFrame:
    df = ctx.study.db
    return df[df.study == study]


# ------------------------------------------------------------------------------- profiles
def fig_profiles(ctx: PlotContext) -> None:
    """Profiles at the baseline fineness and a zoom on the tip with the selected blunting."""
    cfg, R, s = ctx.cfg, ctx.R, ctx.s
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.36 * fs.WIDTH), gridspec_kw=dict(width_ratios=[2.3, 1]))
    f = ctx.opt("profiles", "fineness") or cfg.nose.baseline_fineness
    L = 2 * R * f
    for n in ctx.order():
        c, ls, _ = ctx.sty(n)
        x, y = surface(ctx.p(n), L, R, n=800)
        a.plot(1e3 * x, 1e3 * y, color=c, ls=ls)
    if cfg.payload.enabled:
        rq = cfg.payload.required_radius
        a.axhline(1e3 * rq, color=INK2, lw=0.6, ls=(0, (1, 2)))
        a.text(ctx.opt("profiles", "label_x_mm") * s, 1e3 * rq + ctx.opt("profiles", "label_dy_mm") * s,
               tex(rf"{cfg.payload.name} section needs $r \geq {1e3 * rq:.0f}$ mm"), color=INK2, fontsize=7.5)
    a.set(xlim=(0, 1e3 * L), ylim=(0, ctx.opt("profiles", "ylim_mm") * s), xlabel=r"$x$ from tip [mm]",
          ylabel=r"$r$ [mm]")
    fs.panel(a, "a", y=0.97, x=0.9)
    for n in ctx.order():
        c, ls, _ = ctx.sty(n)
        x, y = surface(ctx.p(n), L, R, n=3000)
        b.plot(1e3 * x, 1e3 * y, color=c, ls=ls, lw=1.0)
    dp = ctx.p(cfg.nose.design_profile)
    rn = cfg.nose.design_tip_radius_ratio * R
    x, y = surface(dp, L, R, rn=rn, n=3000)
    b.fill_between(1e3 * x, 0, 1e3 * y, color=ctx.sty(dp.name)[0], alpha=0.12, lw=0)
    b.plot(1e3 * x, 1e3 * y, color=INK, lw=0.8)
    if rn > 0:
        xt, xc = blunt_tangent_point(dp, L, R, rn)
        tx, ty = ctx.opt("profiles", "tip_label_mm")
        b.annotate(rf"$r_n = {1e3 * rn:g}$ mm", xy=(1e3 * (xc - rn * 0.75), 1e3 * rn * 0.66), xytext=(tx * s, ty * s),
                   fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    b.set(xlim=(0, ctx.opt("profiles", "tip_xlim_mm") * s), ylim=(0, ctx.opt("profiles", "tip_ylim_mm") * s),
          xlabel=r"$x$ [mm]")
    b.set_aspect("equal")
    fs.panel(b, "b", y=0.97, x=0.03)
    shape_legend(ctx, fig, ncol=4, y=0.98)
    ctx.save(fig, "fig_profiles")


# ----------------------------------------------------------------------------- validation
def fig_validation(ctx: PlotContext) -> None:
    """Euler cone pressures against Taylor-Maccoll, and the grid-convergence study."""
    cfg, v = ctx.cfg, ctx.cfg.validation
    d = _db(ctx, v.cone_study)
    dm = _db(ctx, ctx.cfg.aerodynamics.studies[0])
    dm = dm[(dm["shape"] == "cone")]
    if d.empty and dm.empty:
        ctx.log("  validation: no cone data in the database, plot skipped")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    angles = {}
    for f in sorted(d.fineness.unique()):
        angles[np.degrees(np.arctan(1 / (2 * f)))] = d[d.fineness == f]
    if not dm[dm.fineness == v.cone_main_fineness].empty:
        angles[np.degrees(np.arctan(1 / (2 * v.cone_main_fineness)))] = dm[dm.fineness == v.cone_main_fineness]
    seq = list(ctx.opt("validation", "colors"))
    m0, m1 = ctx.opt("validation", "mach_range")
    for i, (thc, dd) in enumerate(sorted(angles.items())):
        c = seq[i % len(seq)]
        Ms = np.linspace(m0, m1, 60)
        cp = []
        for M in Ms:
            try:
                cp.append(cone_cp(M, thc)[0])
            except Exception:
                cp.append(np.nan)
        a.plot(Ms, cp, color=c, lw=1.2)
        a.plot(dd.mach, dd.cd, "o", mfc="white", mec=c, mew=1.0, ms=4.5, ls="none")
        a.text(m1 + 0.02, cp[-1], rf"${thc:.1f}^\circ$", va="center", fontsize=7.5, color=INK)
    a.set(xlim=ctx.opt("validation", "xlim"), xlabel=r"$M_\infty$", ylabel=r"cone surface $C_p$ $(= C_{D,\mathrm{w}})$")
    a.legend(handles=[Line2D([], [], color=INK2, label="Taylor--Maccoll (exact)"),
                      Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="Euler (this work)")],
             loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=7)
    fs.panel(a, "a", x=0.90, y=0.96)
    # grid convergence with Richardson extrapolation
    g = pd.concat([_db(ctx, v.grid_study), _db(ctx, ctx.cfg.aerodynamics.studies[0])])
    g = g[(g["shape"] == v.grid_profile) & (g.fineness == v.grid_fineness) & (g.rn == 0)]
    ncell = {k: GridSpec.from_level(lv).n_cells for k, lv in cfg.cfd.grids.items()}
    for M, (ls, c) in zip(v.grid_machs, ctx.opt("validation", "grid_styles")):
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
        gci = (Fs * abs((f1 - f0) / f1) * 2**p / (2**p - 1) if len(cc) == 3
               else Fs * abs((f1 - f0) / f0) / (2**p - 1)) * 100
        tag = rf"$p={p:.1f}$" if len(cc) == 3 else r"$p=2$ (formal)"
        lab = tex(rf"$M_\infty={M}$: {tag}, GCI$={gci:.1f}\,\%$")
        b.plot(np.concatenate([[0], hh]), np.concatenate([[cex], cc]) / cex, color=c, ls=ls,
               marker="o", ms=4.5, mfc="white", label=lab)
        b.plot(hh[im], cc[im] / cex, "o", color=c, ms=4.5)
    b.axhline(1, color=GRID, lw=0.6, zorder=0)
    b.set(xlim=(-0.001, None), xlabel=r"grid spacing $h = N_{\rm cells}^{-1/2}$",
          ylabel=r"$C_{D,\mathrm{w}} / C_{D,\mathrm{w}}^{\,h\to 0}$")
    if b.get_legend_handles_labels()[0]:
        b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=6.8, handlelength=2.2)
    fs.panel(b, "b", x=0.015)
    fig.tight_layout(w_pad=1.5)
    ctx.save(fig, "fig_validation")


# -------------------------------------------------------------------- wave drag vs M and f
def fig_cdw(ctx: PlotContext) -> None:
    """Nose wave drag versus Mach number (model and CFD points) and versus fineness."""
    cfg, st = ctx.cfg, ctx.study
    f = ctx.opt("cdw", "fineness") or cfg.nose.baseline_fineness
    df = _db(ctx, cfg.aerodynamics.studies[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH))
    m0, m1 = ctx.opt("cdw", "mach_range")
    Mf = np.linspace(m0, m1, 300)
    tr = cfg.aerodynamics
    ylim = ctx.opt("cdw", "ylim")
    a.axvspan(tr.transonic_onset, tr.transonic_match, color=GRID, alpha=0.45, lw=0)
    a.text(round(0.5 * (tr.transonic_onset + tr.transonic_match), 9), 0.968 * ylim[1], "transonic:\nmodelled", ha="center",
           va="top", fontsize=7, color=INK2)
    for n in ctx.order():
        c, ls, mk = ctx.sty(n)
        try:
            w = st.wave_drag(n, f)
        except ValueError:
            continue
        a.plot(Mf, w(Mf), color=c, ls=ls)
        d = df[(df["shape"] == n) & (df.fineness == f)]
        a.plot(d.mach, d.cd, ls="none", marker=mk, color=c, mfc="white", ms=4)
    a.set(xlim=(m0, m1), ylim=ylim, xlabel=r"$M_\infty$", ylabel=rf"nose wave drag $C_{{D,\mathrm{{w}}}}$ ($L/D={f:g}$)")
    fs.panel(a, "a", x=0.03)
    Mb = ctx.opt("cdw", "fineness_mach")
    ff_all = []
    for n in ctx.order():
        c, ls, mk = ctx.sty(n)
        d = df[(df["shape"] == n) & (df.mach == Mb) & (df.fineness <= tr.max_fineness)].sort_values("fineness")
        ff_all += list(d.fineness)
        b.plot(d.fineness, d.cd, color=c, ls=ls, marker=mk, mfc="white", ms=4)
    ff = np.array(sorted(set(ff_all)), dtype=float) if ff_all else np.array([3, 4, 5.0])
    yl = ctx.opt("cdw", "fineness_ylim")
    b.plot(ff, 1 / ff**2, color=INK, lw=0.8, ls=(0, (1, 1)))
    b.annotate(tex("slender-body theory,\n" + r"Von K\'arm\'an: $1/(L/D)^2$"), xy=(ff.min() + 0.5, 1 / (ff.min() + 0.5)**2),
               xytext=(round(ff.min() + 0.04, 9), round(yl[0] * 1.15, 9)), fontsize=7, color=INK, ha="left", va="bottom",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    b.set(xscale="log", yscale="log", ylim=yl, xlabel=r"fineness $L/D$", ylabel=rf"$C_{{D,\mathrm{{w}}}}$ at $M_\infty={Mb:g}$")
    b.set_xticks(list(ff))
    b.set_xticklabels([f"{x:g}" for x in ff])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    yt = [t for t in ctx.opt("cdw", "yticks") if yl[0] <= t <= yl[1]]
    b.set_yticks(yt)
    b.set_yticklabels([f"{t:.2f}" for t in yt])
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fs.panel(b, "b", x=0.90, y=0.96)
    shape_legend(ctx, fig, ncol=4, y=0.98, marker=True)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.98))
    ctx.save(fig, "fig_cdw")


# ------------------------------------------------------------------------- wall pressure
def fig_cp(ctx: PlotContext) -> None:
    """Wall pressure coefficient along the nose (from the saved CFD wall-pressure files)."""
    cfg = ctx.cfg
    f = ctx.opt("cp", "fineness") or cfg.nose.baseline_fineness
    study = cfg.aerodynamics.studies[0]
    cp_dir = cfg.path(cfg.cfd.cp_dir)
    machs = list(ctx.opt("cp", "machs"))
    if not any((cp_dir / f"{study}_{n}_f{f:g}_rn0_{cfg.aerodynamics.grid}_M{M:g}.npz").exists()
               for n in ctx.order() for M in machs):
        ctx.log("  cp: no wall-pressure files for this fineness, plot skipped")
        return
    fig, axs = plt.subplots(1, len(machs), figsize=(fs.WIDTH, 0.38 * fs.WIDTH), sharey=True, squeeze=False)
    xl, yl = ctx.opt("cp", "xlim"), ctx.opt("cp", "ylim")
    for ax, M, let in zip(axs[0], machs, "abcdef"):
        for n in ctx.order():
            fn = cp_dir / f"{study}_{n}_f{f:g}_rn0_{cfg.aerodynamics.grid}_M{M:g}.npz"
            if not fn.exists():
                continue
            c, ls, _ = ctx.sty(n)
            z = np.load(fn)
            x, cp, ish = z["x"], z["cp"], int(z["ish"])
            Lx = x[ish - 1] + 0.5 * (x[ish] - x[ish - 1])
            sel = x < Lx + ctx.opt("cp", "behind_shoulder")
            ax.plot(x[sel] / (2.0 * f), cp[sel], color=c, ls=ls)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(1, color=INK2, lw=0.5, ls=(0, (1, 2)))
        ax.text(1.02, yl[1] - 0.03, "shoulder", fontsize=7, color=INK2, rotation=90, va="top")
        ax.set(xlim=xl, ylim=yl, xlabel=r"$x/L$")
        fs.panel(ax, let, x=0.03, y=0.09)
        ax.text(0.12, 0.055, rf"$M_\infty = {M}$", transform=ax.transAxes, fontsize=8)
    axs[0][0].set_ylabel(r"wall pressure coefficient $C_p$")
    shape_legend(ctx, fig, ncol=4, y=0.98)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.98))
    ctx.save(fig, "fig_cp")


# ---------------------------------------------------------------------------- trajectory
def _traj_label(cfg: Config, mass_model: str, T_ground: Optional[float], kind: str) -> str:
    T = cfg.atmosphere.ground_temperature if T_ground is None else T_ground
    m_bo = burnout_mass(cfg, mass_model)
    cond = "ISA" if abs(T - 288.15) < 1e-9 else rf"${T - 273.15:.0f}^\circ$C"
    return rf"{kind} ({cond}, $m_{{\rm bo}}={m_bo:.0f}$ kg)"


def fig_trajectory(ctx: PlotContext) -> None:
    """Mach number along the ascents and where the drag impulse is spent."""
    cfg = ctx.cfg
    veh, nom, env = ctx.traj
    color = ctx.opt("trajectory", "color")
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    fl = cfg.flight
    for tr, ls, lab, dy in [(nom, "-", _traj_label(cfg, fl.mass_model, None, "nominal"), -0.32),
                            (env, "--", _traj_label(cfg, fl.envelope_mass_model, fl.envelope_ground_temperature,
                                                    "envelope"), -0.08)]:
        a.plot(tr["t"], tr["M"], color=color, ls=ls, label=lab)
        i = np.argmax(tr["M"])
        a.plot(tr["t"][i], tr["M"][i], "o", color=color, ms=4)
        a.annotate(rf"$M={tr['M'][i]:.2f}$ at $z={tr['z'][i]/1e3:.1f}$ km", xy=(tr["t"][i], tr["M"][i]),
                   xytext=(tr["t"][i] + (2.5 if dy < -0.2 else 5.0), tr["M"][i] + dy), fontsize=7,
                   arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    a.axhspan(cfg.aerodynamics.transonic_onset, cfg.aerodynamics.transonic_match, color=GRID, alpha=0.45, lw=0)
    xl = ctx.opt("trajectory", "xlim") or (0, float(np.ceil(max(nom["t"][-1], env["t"][-1]) / 5) * 5))
    yl = ctx.opt("trajectory", "ylim") or (0, float(np.ceil(1.35 * max(nom["M"].max(), env["M"].max()) * 10) / 10))
    a.set(xlim=xl, ylim=yl, xlabel=r"$t$ [s]", ylabel=r"$M$")
    a.legend(loc="upper right", fontsize=7)
    fs.panel(a, "a", x=0.93, y=0.08)
    # where is the drag impulse spent? (nominal flight)
    bw = ctx.opt("trajectory", "bin_width")
    bmax = ctx.opt("trajectory", "mach_bins_max") or float(np.ceil((nom["M"].max() + 0.25) / 0.05) * 0.05)
    bins = np.arange(0, bmax, bw)
    tot, nose = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    t, M = nom["t"], nom["M"]
    dt = np.gradient(t)
    for kk in range(len(t)):
        Re = nom["rho"][kk] * nom["V"][kk] / nom["mu"][kk]
        cdt = veh.cd(M[kk], Re, t[kk] < cfg.engine.burn_time)
        cdn = float(veh.cdw(M[kk]))
        j = np.searchsorted(bins, M[kk]) - 1
        if 0 <= j < len(tot):
            tot[j] += nom["q"][kk] * veh.Aref * cdt * dt[kk]
            nose[j] += nom["q"][kk] * veh.Aref * cdn * dt[kk]
    T = tot.sum()
    nose_name, nose_f = _traj_nose(ctx)
    b.bar(bins[:-1], tot / T * 100, width=bw, align="edge", color="#b7d3f6", edgecolor="white", lw=1,
          label="whole vehicle")
    b.bar(bins[:-1], nose / T * 100, width=bw, align="edge", color="#4a3aa7", edgecolor="white", lw=1,
          label=rf"nose wave drag ({ctx.label(nose_name)}, $L/D={nose_f:g}$)")
    b.set(xlim=(0, bins[-1]), ylim=(0, 1.38 * (tot / T * 100).max()), xlabel=r"$M$",
          ylabel=tex(r"share of drag impulse $\int D\,\mathrm{d}t$ [\%]"))
    b.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 1))
    b.text(0.03, 0.62, tex(rf"nose wave drag: {nose.sum()/T*100:.0f}\,\% of $\int D\,\mathrm{{d}}t$"),
           transform=b.transAxes, ha="left", fontsize=7.5)
    fs.panel(b, "b", x=0.92, y=0.08)
    fig.tight_layout(w_pad=1.5)
    ctx.save(fig, "fig_trajectory")


def _traj_nose(ctx: PlotContext) -> tuple[str, float]:
    if ctx.cfg.heating.trajectory_from == "baseline":
        return ctx.study.baseline
    p, f, _ = ctx.study.design
    return p, f


# ----------------------------------------------------------------------------- breakdown
def fig_breakdown(ctx: PlotContext) -> None:
    """Vehicle drag build-up versus Mach number for the selected nose (coasting)."""
    cfg, st = ctx.cfg, ctx.study
    k = st.k
    p, f, _ = st.design
    veh = st.vehicle(p, f)
    mr = ctx.opt("breakdown", "mach_range") or (0.1, max(cfg.aerodynamics.machs))
    Ms = np.linspace(mr[0], mr[1], 300)
    atm = Atmosphere.from_config(cfg)
    comp = {key: [] for key in ["friction", "base", "fins", "excr", "nose_wave"]}
    for M in Ms:
        T, pr, rho, mu, a = atm(ctx.opt("breakdown", "altitude"))
        c = veh.components(M, rho * M * a / mu, False)
        for key in comp:
            comp[key].append(c[key] * (k if key != "nose_wave" else 1.0))
    labels = {"friction": "skin friction (body + fins)", "base": "base", "fins": "fin pressure",
              "excr": "excrescences", "nose_wave": rf"nose wave drag ({ctx.label(p)}, $L/D={f:g}$)"}
    cols = {"friction": "#2a78d6", "base": "#eb6834", "fins": "#1baf7a", "excr": "#eda100", "nose_wave": "#4a3aa7"}
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.40 * fs.WIDTH))
    ys = [np.array(comp[key]) for key in comp]
    ax.stackplot(Ms, ys, colors=[cols[key] for key in comp], labels=[labels[key] for key in comp],
                 edgecolor="white", linewidth=0.8, alpha=0.9)
    alt = ctx.opt("breakdown", "altitude")
    ax.set(xlim=(mr[0], mr[1]), ylim=(0, None), xlabel=r"$M$", ylabel=rf"$C_D$ (coasting, $z = {alt / 1e3:g}$ km)")
    ax.legend(loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0.0, 1.01), columnspacing=1.0)
    tot = np.sum(ys, axis=0)
    Ma = ctx.opt("breakdown", "annotate_mach")
    i = np.argmin(abs(Ms - Ma))
    ax.annotate(tex(rf"nose: {100 * ys[-1][i] / tot[i]:.0f}\,\% of $C_D$ at $M={Ma:g}$"), xy=(Ma, tot[i]),
                xytext=(Ma - 0.25, tot.max() * 1.05), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    ax.set_ylim(0, tot.max() * 1.15)
    ctx.save(fig, "fig_breakdown")


# --------------------------------------------------------------------------------- trade
def fig_trade(ctx: PlotContext) -> None:
    """Apogee change, payload fit and static-margin change versus fineness for every profile."""
    cfg, t = ctx.cfg, ctx.trade
    pl = cfg.payload.enabled and "l3U_mm" in t
    fig, axs = plt.subplots(1, 3 if pl else 2, figsize=(fs.WIDTH, 0.36 * fs.WIDTH))
    ax_z, ax_sm = axs[0], axs[-1]
    for n in ctx.order():
        d = t[t["shape"] == n].sort_values("fineness")
        if "tip_radius_ratio" in d:
            d = d[np.isclose(d.tip_radius_ratio, d.tip_radius_ratio.min())]
        c, ls, mk = ctx.sty(n)
        kw = dict(color=c, ls=ls, marker=mk, mfc="white", ms=3.5)
        ax_z.plot(d.fineness, d.d_apogee, **kw)
        if pl:
            axs[1].plot(d.fineness, d.l3U_mm, **kw)
        ax_sm.plot(d.fineness, d.dSM_cal, **kw)
    ax_z.set(ylabel=r"$\Delta z_{\rm apogee}$ vs.\ baseline [m]")
    fmin = float(t.fineness.min())
    if pl:
        Lp = 1e3 * cfg.payload.length
        axs[1].axhline(Lp, color=INK2, lw=0.6, ls=(0, (1, 2)))
        axs[1].text(fmin + 0.05, Lp + 6 * ctx.s, f"full {cfg.payload.name} length", fontsize=7, color=INK2)
        axs[1].set(ylabel=rf"length with $r\geq {1e3 * cfg.payload.required_radius:.0f}$ mm [mm]",
                   ylim=ctx.opt("trade", "payload_ylim") or (0, 1.5 * Lp))
    ax_sm.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax_sm.set(ylabel=r"$\Delta$ static margin [cal]")
    for ax, let in zip(axs, "abc"):
        ax.set_xlabel(r"fineness $L/D$")
        ax.plot(cfg.nose.baseline_fineness, 0 if (not pl or ax is not axs[1]) else np.nan, "x", color=INK, ms=5)
        fs.panel(ax, let, x=0.04)
    shape_legend(ctx, fig, ncol=4, y=0.98, marker=True)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.98))
    ctx.save(fig, "fig_trade")


# ---------------------------------------------------------------------------- robustness
def fig_robustness(ctx: PlotContext) -> None:
    """Apogee of every profile relative to the reference profile, for every scenario."""
    cfg, d = ctx.cfg, ctx.robustness
    order = [n for n in ctx.order() if n in set(d["shape"])]
    scen = list(dict.fromkeys(d.scenario))
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.46 * fs.WIDTH))
    marks, seq = ctx.opt("robustness", "markers"), ctx.opt("robustness", "colors")
    for j, sc in enumerate(scen):
        dd = d[d.scenario == sc].set_index("shape").reindex(order)
        ypos = np.arange(len(order)) + (j - (len(scen) - 1) / 2) * 0.09
        ax.plot(dd.dz_vs_vk, ypos, ls="none", marker=marks[j % len(marks)], ms=4.2 if j else 5,
                mfc=seq[j % len(seq)] if j == 0 else "white", mec=seq[j % len(seq)], mew=1.0,
                label=tex(sc.replace("%", r"\%").replace("30 C", r"30$^\circ$C")))
    band = cfg.robustness.band
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvspan(-band, band, color=GRID, alpha=0.4, lw=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ctx.label(n) for n in order])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.invert_yaxis()
    ax.set_xlabel(rf"apogee relative to {ctx.label(cfg.robustness.reference_profile)} at $L/D={cfg.robustness.fineness:g}$ [m]")
    ax.legend(loc="lower left", fontsize=6.8, ncol=1, bbox_to_anchor=(0.0, 0.0), frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.text(0, -0.75, rf"$\pm{band:g}$ m", ha="center", fontsize=7, color=INK2)
    ctx.save(fig, "fig_robustness")


# ----------------------------------------------------------------------------- bluntness
def fig_bluntness(ctx: PlotContext) -> None:
    """Drag cost of a spherical tip and stagnation-point heat transfer versus tip radius."""
    cfg, bl = ctx.cfg, ctx.cfg.bluntness
    _, nom, env = ctx.traj
    d = pd.concat([_db(ctx, bl.study), _db(ctx, cfg.aerodynamics.studies[0])])
    d = d[(d["shape"] == bl.profile) & (d.fineness == bl.fineness) & (d.grid == cfg.aerodynamics.grid)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.38 * fs.WIDTH))
    for M, c, ls in zip(ctx.opt("bluntness", "machs"), ctx.opt("bluntness", "colors"), ctx.opt("bluntness", "styles")):
        dd = d[d.mach == M].sort_values("rn").drop_duplicates("rn")
        if dd.empty:
            continue
        a.plot(dd.rn, dd.cd / dd.cd.values[0], color=c, ls=ls, marker="o", mfc="white", ms=4, label=rf"$M_\infty = {M}$")
    a.axhline(1, color=GRID, lw=0.6, zorder=0)
    a.axvline(cfg.nose.design_tip_radius_ratio, color=INK2, lw=0.5, ls=(0, (1, 2)))
    a.set(xlabel=r"bluntness $r_n/R$", ylabel=r"$C_{D,\mathrm{w}}(r_n) / C_{D,\mathrm{w}}(r_n=0)$")
    if a.get_legend_handles_labels()[0]:
        a.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.05, 0.93))
    a.text(0.97, 0.06, rf"{ctx.label(bl.profile)}, $L/D={bl.fineness:g}$", transform=a.transAxes, ha="right", fontsize=7.5)
    fs.panel(a, "a", x=0.03)
    r0, r1 = bl.radius_range
    rn = np.linspace(r0, r1, 60)
    hh = {}
    for tr, ls, lab in [(nom, "-", "nominal"), (env, "--", "envelope")]:
        i = np.argmax(tr["M"])
        h = np.array([ht.h_stagnation(tr["M"][i], tr["p"][i], tr["T"][i], r)[0] for r in rn])
        hh[lab] = h
        b.plot(1e3 * rn, h / 1e3, color="#2a78d6", ls=ls, label=lab)
    guide = ctx.opt("bluntness", "guide")
    if guide is None:
        r_ref = 0.5 * (r0 + r1)
        guide = (1.05 * float(np.interp(r_ref, rn, hh["envelope"])) / 1e3, r_ref)
    c_g, r_g = guide
    b.plot(1e3 * rn, c_g * (r_g / rn) ** 0.5, color=INK, lw=0.6, ls=(0, (1, 1.5)))
    r_lab = ctx.opt("bluntness", "guide_label_r") or 0.7 * r1
    b.text(1e3 * r_lab, c_g * (r_g / r_lab) ** 0.5 * 1.08, r"$\propto r_n^{-1/2}$", fontsize=7.5)
    b.axvline(1e3 * cfg.nose.design_tip_radius_ratio * ctx.R, color=INK2, lw=0.5, ls=(0, (1, 2)))
    b.set(xlabel=r"tip radius $r_n$ [mm]", ylabel=r"$h_s$ at $M_{\max}$ [kW/(m$^2$K)]")
    b.legend(loc="upper right", fontsize=7)
    fs.panel(b, "b", x=0.03)
    fig.tight_layout(w_pad=1.5)
    ctx.save(fig, "fig_bluntness")


# ------------------------------------------------------------------------------- thermal
def _station_label(s, L: float) -> str:
    if isinstance(s, str) and s.strip().endswith("L"):
        fr = float(s.strip()[:-1])
        return r"$x=L$ (shoulder)" if fr == 1.0 else (r"$x=L/2$" if fr == 0.5 else rf"$x={fr:g}L$")
    return rf"$x={1e3 * parse_station(s, L):g}$ mm"


def fig_thermal(ctx: PlotContext) -> None:
    """Inner-face temperature of the shell and temperature of the tip (printed vs metal)."""
    cfg, st, h = ctx.cfg, ctx.study, ctx.cfg.heating
    _, nom, env = ctx.traj
    p, f, rnr = st.design
    L = st.L(f)
    mats = st.materials()
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    cols = ctx.opt("thermal", "colors")
    stations = [(parse_station(s, L), cols[i % len(cols)], _station_label(s, L)) for i, s in enumerate(h.shell_stations)]
    tmax = []
    for tr, ls in [(nom, "-"), (env, "--")]:
        for x, c, lab in stations:
            r = ht.wall_response(tr, mats[h.shell_material], h.shell_thickness, x, n=h.wall_nodes)
            a.plot(r["t"], r["T_in"] - 273.15, color=c, ls=ls)
        p0, T0 = ht.stagnation_state(tr["M"], tr["p"], tr["T"])
        a.plot(tr["t"], T0 - 273.15, color=INK, lw=0.7, ls=ls)
        tmax.append(float(np.max(T0)) - 273.15)
    xl = ctx.opt("thermal", "xlim") or (0, float(np.ceil(1.6 * cfg.engine.burn_time * 2 / 10) * 10))
    yl = ctx.opt("thermal", "ylim") or (0, float(np.ceil(1.12 * max(max(tmax), h.bond_limit) / 25) * 25))
    a.axhline(h.bond_limit, color="#d03b3b", lw=0.7, ls=(0, (4, 2)))
    a.text(xl[1] - 0.5, h.bond_limit + 3, h.bond_label, fontsize=7, color=INK, ha="right")
    a.set(xlim=xl, ylim=yl, xlabel=r"$t$ [s]", ylabel=r"temperature [$^\circ$C]")
    a.legend(handles=[Line2D([], [], color=c, label=lab) for _, c, lab in stations]
             + [Line2D([], [], color=INK2, ls="-", label="nominal"), Line2D([], [], color=INK2, ls="--", label="envelope")],
             loc="lower left", fontsize=6.8, ncol=3, bbox_to_anchor=(0, 1.01), columnspacing=1.0)
    fs.panel(a, "a", x=0.03)
    rn = rnr * ctx.R
    pr = ctx.p(p)
    for tr, ls in [(nom, "-"), (env, "--")]:
        r1 = ht.wall_response(tr, mats[h.printed_tip_material], h.printed_tip_thickness, "stagnation", rn=rn, n=h.wall_nodes)
        r2 = ht.tip_insert(tr, pr, L, ctx.R, rn, h.tip_length, cfg.structure.tip_mass, mats[h.tip_material])
        b.plot(r1["t"], r1["T_out"] - 273.15, color="#eb6834", ls=ls)
        b.plot(r2["t"], r2["T"] - 273.15, color="#2a78d6", ls=ls)
        p0, T0 = ht.stagnation_state(tr["M"], tr["p"], tr["T"])
        b.plot(tr["t"], T0 - 273.15, color=INK, lw=0.7, ls=ls)
    lo, hi = h.printed_tip_hdt
    b.axhspan(lo, hi, color="#eb6834", alpha=0.15, lw=0)
    b.text(xl[1] - 0.5, hi + 3, f"{mats[h.printed_tip_material].label} HDT range", fontsize=7, ha="right", color=INK)
    b.legend(handles=[Line2D([], [], color="#eb6834", label=f"{mats[h.printed_tip_material].label} printed tip (surface)"),
                      Line2D([], [], color="#2a78d6", label=f"{mats[h.tip_material].label} machined tip "
                                                            f"({1e3 * cfg.structure.tip_mass:.0f} g)")],
             loc="lower left", fontsize=6.8, bbox_to_anchor=(0, 1.01), ncol=1)
    b.set(xlim=xl, xlabel=r"$t$ [s]")
    fs.panel(b, "b", x=0.03)
    fig.tight_layout(w_pad=1.0)
    ctx.save(fig, "fig_thermal")


# --------------------------------------------------------------------------- final design
def fig_final(ctx: PlotContext) -> None:
    """Selected nose to scale with the tip insert, shoulder and payload envelope."""
    cfg, R, sc = ctx.cfg, ctx.R, ctx.s
    p, f, rnr = ctx.study.design
    pr = ctx.p(p)
    L = 2 * R * f
    rn = rnr * R
    x, y = surface(pr, L, R, rn=rn, n=1500)
    x0 = x[0]
    fig, a = plt.subplots(figsize=(fs.WIDTH, 0.30 * fs.WIDTH))
    X, Y = 1e3 * (x - x0), 1e3 * y
    sh = 1e3 * cfg.nose.shoulder_length
    Rm = 1e3 * R
    for sgn in (1, -1):
        a.plot(X, sgn * Y, color=INK, lw=1.0)
        a.plot([X[-1], X[-1] + sh], [sgn * Rm, sgn * Rm], color=INK, lw=1.0)
    a.plot([X[-1] + sh] * 2, [-Rm, Rm], color=INK, lw=1.0)
    a.plot([X[-1]] * 2, [-Rm, Rm], color=INK2, lw=0.5, ls=(0, (3, 2)))
    mats = ctx.study.materials()
    x_tip = 1e3 * cfg.heating.tip_length
    sel = X <= x_tip
    a.fill_between(X[sel], -Y[sel], Y[sel], color="#86b6ef", lw=0)
    a.annotate(f"{mats[cfg.heating.tip_material].label} tip insert", xy=(x_tip * 0.6, -8 * sc),
               xytext=(x_tip * 0.6, -80 * sc), fontsize=6.5, ha="center",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    if cfg.payload.enabled:
        l3 = 1e3 * cfg.payload.length
        w3 = 1e3 * cfg.payload.width
        x3 = X[-1] + sh - l3 - 5 * sc
        a.add_patch(plt.Rectangle((x3, -w3 / 2), l3, w3, fill=False, ec="#eb6834", lw=0.8, ls=(0, (4, 2))))
        a.text(x3 + l3 / 2, 0, f"{cfg.payload.name} CubeSat envelope" if cfg.payload.name == "3U"
               else f"{cfg.payload.name} envelope", fontsize=6.5, ha="center", va="center", color=INK)
    a.annotate("", xy=(0, Rm + 15 * sc), xytext=(X[-1], Rm + 15 * sc), arrowprops=dict(arrowstyle="<->", lw=0.6))
    a.text(X[-1] / 2, Rm + 20 * sc, rf"{X[-1]:.0f} mm tip to shoulder (profile length $L = {1e3 * L:.0f}$ mm)",
           ha="center", fontsize=7.5)
    a.annotate("", xy=(X[-1] + sh + 15 * sc, -Rm), xytext=(X[-1] + sh + 15 * sc, Rm), arrowprops=dict(arrowstyle="<->", lw=0.6))
    a.text(X[-1] + sh + 22 * sc, 0, rf"$\varnothing {2 * Rm:.0f}$", rotation=90, va="center", fontsize=7.5)
    a.text(X[-1] + sh / 2, -Rm - 10 * sc, rf"shoulder {sh:.0f} mm", ha="center", va="top", fontsize=6.5)
    a.set_aspect("equal")
    a.axis("off")
    a.set(xlim=(-10 * sc, X[-1] + sh + 40 * sc), ylim=(-Rm - 22 * sc, Rm + 32 * sc))
    ctx.save(fig, "fig_final")


# --------------------------------------------------------------------------------- gallery
def shaded_body(ax: Any, profile: Profile, L: float, R: float, sh: float, color: str, rn: float = 0.0,
                tip_x: Optional[float] = None, res: float = 0.6) -> None:
    """Side view of the solid of revolution with Lambert + Phong shading (to scale, mm)."""
    from matplotlib.colors import to_rgb
    xs = np.arange(0, 1e3 * (L + sh) + res, res)
    ys = np.arange(-1e3 * R - res, 1e3 * R + res, res)
    xb, yb = surface(profile, L, R, rn=rn, n=3000)
    x0 = xb[0]
    xb = 1e3 * (xb - x0)
    yb = 1e3 * yb
    r = np.interp(xs, xb, yb, left=0.0, right=1e3 * R)
    dr = np.gradient(r, xs)
    Xg, Yg = np.meshgrid(xs, ys)
    rg = r[None, :]
    inside = np.abs(Yg) < rg
    sphi = np.clip(Yg / np.maximum(rg, 1e-9), -1, 1)
    cphi = np.sqrt(1 - sphi**2)
    n = np.stack([-np.broadcast_to(dr[None, :], Yg.shape), sphi, cphi], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Ld = np.array([-0.35, 0.55, 0.76])
    Ld /= np.linalg.norm(Ld)
    H = Ld + np.array([0, 0, 1.0])
    H /= np.linalg.norm(H)
    diff = np.clip(n @ Ld, 0, 1)
    spec = np.clip(n @ H, 0, 1) ** 40
    base = np.array(to_rgb(color)) * 0.65 + 0.35
    img = np.ones(Yg.shape + (4,))
    rgb = base[None, None, :] * (0.32 + 0.68 * diff[..., None]) + 0.35 * spec[..., None]
    if tip_x is not None:
        grey = np.array(to_rgb("#a7b0b8"))
        mask = Xg <= tip_x
        rgb[mask] = grey * (0.32 + 0.68 * diff[mask][:, None]) + 0.45 * spec[mask][:, None]
    img[..., :3] = np.clip(rgb, 0, 1)
    img[..., 3] = inside.astype(float)
    ax.imshow(img, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", interpolation="bilinear")
    ax.plot(np.concatenate([xb, [1e3 * (L + sh) - 1e3 * x0]]), np.concatenate([yb, [1e3 * R]]), color=INK, lw=0.4)
    ax.plot(np.concatenate([xb, [1e3 * (L + sh) - 1e3 * x0]]), -np.concatenate([yb, [1e3 * R]]), color=INK, lw=0.4)
    ax.plot([1e3 * (L + sh) - 1e3 * x0] * 2, [-1e3 * R, 1e3 * R], color=INK, lw=0.4)
    ax.plot([1e3 * (L - x0)] * 2, [-1e3 * R, 1e3 * R], color=INK2, lw=0.35, ls=(0, (2, 2)))


def _family_text(pr: Profile) -> str:
    fam, p = pr.family, pr.p
    if fam == "cone":
        return r"$\eta=\xi$"
    if fam == "tangent_ogive":
        return r"circular arc, $\rho=(R^2+L^2)/2R$"
    if fam == "parabolic":
        return r"$\eta=2\xi-\xi^2$" if p["K"] == 1.0 else rf"$\eta=(2\xi-{p['K']:g}\xi^2)/(2-{p['K']:g})$"
    if fam == "power":
        n = p["n"]
        known = {0.5: "1/2", 0.75: "3/4", 0.25: "1/4"}
        return rf"$\eta=\xi^{{{known.get(n, f'{n:g}')}}}$"
    if fam == "haack":
        Cv = p["C"]
        cs = r"\tfrac13" if abs(Cv - 1 / 3) < 1e-12 else f"{Cv:g}"
        return rf"Haack, $C={cs}$"
    return tex(pr.long_label or pr.name)


def fig_gallery(ctx: PlotContext) -> None:
    """All profiles at the design fineness, to scale, with the payload envelope and key numbers."""
    cfg, R, sc = ctx.cfg, ctx.R, ctx.s
    f = cfg.nose.design_fineness
    t = ctx.trade[np.isclose(ctx.trade.fineness, f)]
    if "tip_radius_ratio" in t:
        t = t[np.isclose(t.tip_radius_ratio, t.tip_radius_ratio.min())]
    t = t.set_index("shape")
    order = [n for n in ctx.order() if n in t.index]
    L = 2 * R * f
    sh = cfg.nose.shoulder_length
    fig, axs = plt.subplots(len(order), 2, figsize=(fs.WIDTH, 1.02 * fs.WIDTH * (len(order) / 7)),
                            gridspec_kw=dict(width_ratios=[2.15, 1], hspace=0.12, wspace=0.02), squeeze=False)
    Rm = 1e3 * R
    for row, n in enumerate(order):
        a, b = axs[row]
        c = ctx.sty(n)[0]
        shaded_body(a, ctx.p(n), L, R, sh, c)
        if cfg.payload.enabled and "l3U_mm" in t:
            l3 = 1e3 * cfg.payload.length
            w3 = 1e3 * cfg.payload.width
            x3_front = 1e3 * (L - t.loc[n, "l3U_mm"] / 1e3)
            a.add_patch(plt.Rectangle((x3_front, -w3 / 2), l3, w3, fill=False, ec=INK, lw=0.6, ls=(0, (3, 2))))
            x_end = max(1e3 * (L + sh), x3_front + l3 + 10 * sc)
            if x_end > 1e3 * (L + sh):   # payload-bay cylinder needed behind the shoulder
                for yy in (Rm, -Rm):
                    a.plot([1e3 * (L + sh), x_end], [yy, yy], color=INK2, lw=0.5, ls=(0, (2, 2)))
                a.plot([x_end, x_end], [-Rm, Rm], color=INK2, lw=0.5, ls=(0, (2, 2)))
        a.set_xlim(-10 * sc, 1e3 * (L + sh) + 300 * sc)
        a.set_ylim(-Rm - 10 * sc, Rm + 32 * sc)
        a.set_aspect("equal")
        a.axis("off")
        title = ctx.label(n) + (" (recommended)" if n == cfg.nose.design_profile else "")
        a.text(-8 * sc, Rm + 6 * sc, tex(rf"\textbf{{{title}}}"), fontsize=8, ha="left", va="bottom",
               color=c if n == cfg.nose.design_profile else INK,
               **({"fontweight": "bold"} if not fs.USETEX else {}))
        r = t.loc[n]
        Mg = ctx.opt("gallery", "mach")
        col = f"cd_w_{mach_key(Mg)}"
        lines = [tex(_family_text(ctx.p(n))),
                 rf"$C_{{D,\mathrm{{w}}}}({Mg:g})={r[col]:.4f}$, $\Delta z={r.d_apogee:+.0f}$ m"
                 if col in r else rf"$\Delta z={r.d_apogee:+.0f}$ m"]
        if cfg.payload.enabled and "l3U_mm" in r:
            lines.append(rf"mass {r.mass_g:.0f} g, {cfg.payload.name} in nose {r.l3U_mm:.0f} mm")
        else:
            lines.append(rf"mass {r.mass_g:.0f} g")
        lines.append(rf"shoulder ${base_angle_deg(ctx.p(n), L, R):.1f}^\circ$, $x_{{cp}}/L={r.xcp_over_L:.2f}$")
        b.text(0.0, 0.5, "\n".join(lines), fontsize=6.8, va="center", ha="left", transform=b.transAxes, color=INK,
               linespacing=1.35)
        b.axis("off")
    ctx.save(fig, "fig_gallery")


# ----------------------------------------------------------------------------------- fields
def fig_fields(ctx: PlotContext) -> None:
    """Pressure-coefficient field around every profile (saved Euler flow fields)."""
    cfg = ctx.cfg
    fdir = cfg.path(cfg.cfd.fields_dir)
    f = cfg.nose.design_fineness
    for M in ctx.opt("fields", "machs"):
        order = [n for n in ctx.order() if (fdir / f"{n}_rn0_M{M:g}.npz").exists()]
        if not order:
            ctx.log(f"  fields: no saved flow fields at M = {M:g}, plot skipped")
            continue
        nrow = int(np.ceil((len(ctx.order()) + 1) / 2))
        fig, axs = plt.subplots(nrow, 2, figsize=(fs.WIDTH, 0.80 * fs.WIDTH * (nrow / 4)))
        axs = axs.ravel()
        lo, mid, hi = ctx.opt("fields", "clim")
        norm = matplotlib.colors.TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)
        DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f4a3a2", "#c9302f"])
        pc = None
        xl = ctx.opt("fields", "xlim") or (-1.5, 2 * f + 4)
        for k, n in enumerate(ctx.order()):
            ax = axs[k]
            fn = fdir / f"{n}_rn0_M{M:g}.npz"
            if not fn.exists():
                ax.axis("off")
                continue
            z = np.load(fn)
            X, Y, W = z["X"], z["Y"], z["W"].astype(float)
            c = np.sqrt(1.4 * W[..., 3] / W[..., 0])
            Mach = np.hypot(W[..., 1], W[..., 2]) / c
            Cp = (W[..., 3] - 1 / 1.4) / (0.5 * M**2)
            for sgn in (1, -1):
                pc = ax.pcolormesh(X, sgn * Y, Cp, cmap=DIV, norm=norm, shading="flat", rasterized=True)
            xb, yb = X[:, 0], Y[:, 0]
            ax.fill(np.concatenate([xb, xb[::-1]]), np.concatenate([yb, -yb[::-1]]), color="#d9d8d4", lw=0)
            ax.plot(xb, yb, color=INK, lw=0.5)
            ax.plot(xb, -yb, color=INK, lw=0.5)
            xc = 0.25 * (X[1:, 1:] + X[:-1, 1:] + X[1:, :-1] + X[:-1, :-1])
            yc = 0.25 * (Y[1:, 1:] + Y[:-1, 1:] + Y[1:, :-1] + Y[:-1, :-1])
            ax.tricontour(xc.ravel(), yc.ravel(), Mach.ravel(), levels=[1.0], colors=[INK], linewidths=0.5,
                          linestyles=[(0, (2, 1.5))])
            ax.set_xlim(*xl)
            ax.set_ylim(*ctx.opt("fields", "ylim"))
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            ax.text(0.015, 0.94, ctx.label(n), transform=ax.transAxes, fontsize=7.5, va="top", color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
        for k in range(len(ctx.order()), len(axs) - 1):
            axs[k].axis("off")
        ax = axs[-1]
        ax.axis("off")
        cax = ax.inset_axes([0.08, 0.62, 0.84, 0.11])
        cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
        cb.set_label(rf"pressure coefficient $C_p$ ($M_\infty = {M}$, $L/D = {f:g}$)", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.text(0.5, 0.18, "red: compression, blue: expansion\ndashed: sonic line ($M=1$)",
                transform=ax.transAxes, ha="center", va="center", fontsize=7, color=INK2)
        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        ctx.save(fig, f"fig_fields_M{M:g}".replace(".", ""))


# -------------------------------------------------------------------------------------- grid
def fig_grid(ctx: PlotContext) -> None:
    """The computational grid (coarse level) with its boundaries labelled."""
    cfg = ctx.cfg
    p, f, _ = ctx.study.design
    lvl = cfg.cfd.grids[ctx.opt("grid", "level")]
    X, Y, ish = make_grid(ctx.p(p), f, spec=GridSpec.from_level(lvl, cfg.cfd.cylinder_length, cfg.cfd.upstream_distance,
                                                                 cfg.cfd.outer_angle))
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.42 * fs.WIDTH), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax in (a, b):
        ax.plot(X, Y, color="#9ec5f4", lw=0.35)
        ax.plot(X.T, Y.T, color="#9ec5f4", lw=0.35)
        xb, yb = X[:, 0], Y[:, 0]
        ax.fill_between(xb, 0, yb, color="#d9d8d4", lw=0)
        ax.plot(xb, yb, color=INK, lw=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    a.plot(X[:, -1], Y[:, -1], color="#eb6834", lw=1.4)
    a.plot(X[0, :], Y[0, :], color="#1baf7a", lw=1.4)
    a.plot(X[-1, :], Y[-1, :], color="#4a3aa7", lw=1.4)
    a.set_xlim(X.min() - 0.5, X.max() + 0.5)
    yl = ctx.opt("grid", "ylim")
    a.set_ylim(*yl)
    a.text(-4.0 * f / 4.5, 9 * yl[1] / 22,
           "far field\n(free stream\ncomes in)", color="#eb6834", fontsize=7, ha="center")
    a.text(-1.8 * f / 4.5, 0.5, "axis", color="#1baf7a", fontsize=7, ha="right")
    a.text(X.max() - 0.3, 12 * yl[1] / 22, "outflow", color="#4a3aa7", fontsize=7, ha="right", rotation=90)
    a.annotate("nose + cylinder\n(slip wall)", xy=(6 * f / 4.5, 0.9), xytext=(8 * f / 4.5, 6 * yl[1] / 22), fontsize=7,
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    fs.panel(a, "a", x=0.02, y=0.96)
    b.set_xlim(*ctx.opt("grid", "zoom_xlim"))
    b.set_ylim(*ctx.opt("grid", "zoom_ylim"))
    b.text(1.0, 0.12, "body", fontsize=7, ha="center")
    b.text(0.95, 0.9, "each quadrilateral\n= one cell", fontsize=7, ha="center", va="top",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    fs.panel(b, "b", x=0.03, y=0.96)
    fig.tight_layout(w_pad=1.0)
    ctx.save(fig, "fig_grid")


# -------------------------------------------------------------------------------- schematics
def _schematic_title(pr: Profile) -> str:
    fam, p = pr.family, pr.p
    if fam == "cone":
        return r"Conical: $\eta=\xi$"
    if fam == "tangent_ogive":
        return "Tangent ogive: circular arc"
    if fam == "parabolic":
        return (r"Parabolic ($K'=1$): $\eta=2\xi-\xi^2$" if p["K"] == 1.0
                else rf"Parabolic ($K'={p['K']:g}$)")
    if fam == "power":
        n = p["n"]
        frac = {0.5: (r"\tfrac12", "1/2"), 0.75: (r"\tfrac34", "3/4")}.get(n, (f"{n:g}", f"{n:g}"))
        return rf"Power series, $n={frac[0]}$: $\eta=\xi^{{{frac[1]}}}$"
    if fam == "haack":
        Cv = p["C"]
        name = r"Von K\'arm\'an" if Cv == 0 else ("LV-Haack" if abs(Cv - 1 / 3) < 1e-12 else pr.label)
        cs = r"\tfrac13" if abs(Cv - 1 / 3) < 1e-12 else f"{Cv:g}"
        return rf"{name}: Haack series, $C={cs}$"
    return pr.long_label or pr.name


def fig_schematics(ctx: PlotContext) -> None:
    """Annotated schematics of the profiles, drawn at a low fineness so that the differences show."""
    from matplotlib.patches import Arc, Rectangle
    fd = ctx.opt("schematics", "fineness")
    order = ctx.order()
    Rs, Ls, stub = 1.0, 2.0 * fd, 0.9
    XL, YL = (-0.50, Ls + stub + 2.10), (-1.78, 1.78)
    xc = 0.5 * (XL[0] + XL[1])
    zx = 0.30
    win = (-0.1 * zx, 1.1 * zx)
    ins = (Ls + stub + 0.50, -0.78, 1.55, 1.55)
    y_top, y_bot = 1.47, -1.50
    kw_dim = dict(arrowstyle="<->", lw=0.6, color=INK2, shrinkA=0, shrinkB=0)
    kw_ptr = dict(arrowstyle="-|>", lw=0.5, color=INK2, shrinkA=1, shrinkB=1, mutation_scale=6)
    small = dict(fontsize=7, color=INK2)

    def draw(ax, pr, color, alpha=0.20):
        xx = Ls * np.linspace(0, 1, 1200) ** 3
        yy = radius(pr, xx, Ls, Rs)
        X, Y = np.append(xx, Ls + stub), np.append(yy, Rs)
        ax.fill_between(X, -Y, Y, color=color, alpha=alpha, lw=0)
        for sgn in (1, -1):
            ax.plot(X, sgn * Y, color=color, lw=1.1, solid_capstyle="butt")
        ax.plot([Ls, Ls], [-Rs, Rs], color=color, lw=0.5, ls=(0, (2, 2)))
        ax.plot([XL[0] + 0.12, Ls + stub + 0.15], [0, 0], color=INK2, lw=0.4, ls=(0, (8, 2, 1, 2)))
        ax.set(xlim=XL, ylim=YL, aspect="equal")
        ax.axis("off")

    def tip_inset(ax, pr, color):
        a = ax.inset_axes(ins, transform=ax.transData)
        xx = zx * np.linspace(0, 1, 800) ** 3
        yy = radius(pr, xx, Ls, Rs)
        a.fill_between(xx, -yy, yy, color=color, alpha=0.20, lw=0)
        a.plot(xx, yy, color=color, lw=1.0)
        a.plot(xx, -yy, color=color, lw=1.0)
        a.plot(win, [0, 0], color=INK2, lw=0.4, ls=(0, (8, 2, 1, 2)))
        h = 0.5 * (win[1] - win[0])
        a.set(xlim=win, ylim=(-h, h), xticks=[], yticks=[])
        a.minorticks_off()
        for sp in a.spines.values():
            sp.set(lw=0.5, color=INK2)
        a.text(0.5, 1.04, r"tip region, $\times%d$" % round(ins[2] / (win[1] - win[0])),
               transform=a.transAxes, ha="center", va="bottom", **small)
        ax.add_patch(Rectangle((win[0], -h), 2 * h, 2 * h, fill=False, lw=0.5, ec=INK2))

    def kink(ax, pr, color, is_cone):
        th = np.arctan(slope(pr, Ls, Ls, Rs))
        e = 0.85
        ax.plot([Ls, Ls + e], [Rs, Rs + e * np.tan(th)], color=color, lw=0.7, ls=(0, (2, 1.5)))
        ax.add_patch(Arc((Ls, Rs), 1.3, 1.3, theta1=0, theta2=np.degrees(th), lw=0.6, color=INK))
        lab = (r"shoulder kink $\theta_{\rm sh}=\theta_c$" if is_cone
               else r"shoulder kink $\theta_{\rm sh}=\arctan(nR/L)$" if pr.family == "power"
               else rf"shoulder kink $\theta_{{\rm sh}}={np.degrees(th):.1f}^\circ$")
        ax.annotate(lab, xy=(Ls + 0.55, Rs + 0.07), xytext=(Ls + 0.95, y_top), ha="right",
                    va="center", arrowprops=kw_ptr, fontsize=7, color=INK)

    def tangent(ax):
        ax.annotate("tangent to the body at the shoulder", xy=(Ls, Rs), xytext=(Ls + 0.95, y_top),
                    ha="right", va="center", arrowprops=kw_ptr, **small)

    def bottom(ax, *lines):
        for i, s_ in enumerate(lines):
            ax.text(xc, y_bot + 0.17 * (len(lines) - 1) - 0.34 * i, tex(s_), ha="center", va="center", **small)

    nrow = int(np.ceil((len(order) + 1) / 2))
    fig, axs = plt.subplots(nrow, 2, figsize=(fs.WIDTH, 1.08 * fs.WIDTH * (nrow / 4)))
    axs = axs.ravel()
    ax = axs[0]
    vk = Profile("vk", "haack", (("C", 0.0),))
    draw(ax, vk, INK2, alpha=0.12)
    ax.set_title(tex(r"\textbf{(a)} Notation"), loc="left", fontsize=8.5, pad=2)
    ax.annotate("", xy=(0, 1.32), xytext=(Ls, 1.32), arrowprops=kw_dim)
    ax.plot([0, 0], [0.06, 1.44], color=INK2, lw=0.4)
    ax.plot([Ls, Ls], [Rs + 0.04, 1.44], color=INK2, lw=0.4)
    ax.text(Ls / 2, 1.40, r"$L$", ha="center", va="bottom", fontsize=8)
    xd = Ls + 0.55 * stub
    ax.annotate("", xy=(xd, -Rs), xytext=(xd, Rs), arrowprops=kw_dim)
    ax.text(xd + 0.10, 0.0, r"$D=2R$", ha="left", va="center", rotation=90, fontsize=8)
    xs = 0.40 * Ls
    ys = radius(vk, xs, Ls, Rs)
    ax.annotate("", xy=(xs, ys), xytext=(xs, 0), arrowprops=kw_dim)
    ax.text(xs + 0.08, 0.32, r"$y(x)$", ha="left", va="center", fontsize=8)
    ax.annotate("", xy=(xs, -0.20), xytext=(0, -0.20), arrowprops=dict(kw_dim, arrowstyle="->"))
    ax.text(xs / 2, -0.27, r"$x$", ha="center", va="top", fontsize=8)
    ax.annotate("apex", xy=(0, 0.02), xytext=(-0.28, 0.78), ha="center", va="bottom", arrowprops=kw_ptr, **small)
    ax.annotate("shoulder", xy=(Ls, -Rs), xytext=(Ls - 0.75, -1.45), ha="center", va="center",
                arrowprops=kw_ptr, **small)
    ax.text(Ls + stub / 2, -1.45, "body tube", ha="left", va="center", **small)
    ax.annotate("", xy=(0.45, -1.05), xytext=(-0.48, -1.05), arrowprops=dict(kw_ptr, mutation_scale=7))
    ax.text(-0.48, -0.92, r"$V_\infty$", ha="left", va="bottom", fontsize=8)
    tx = Ls + stub + 0.45
    for i, s_ in enumerate([r"$\xi = x/L$", r"$\eta = y/R$", r"fineness $L/D$", r"$\theta_{\rm sh}$: shoulder angle"]):
        ax.text(tx, 0.62 - 0.36 * i, s_, ha="left", va="center", fontsize=7.5, color=INK)
    letters = "bcdefghijklmnopqrstuvwxyz"
    for k, name in enumerate(order):
        ax, c = axs[k + 1], ctx.sty(name)[0]
        pr = ctx.p(name)
        draw(ax, pr, c)
        ax.set_title(tex(rf"\textbf{{({letters[k]})}} " + _schematic_title(pr)), loc="left", fontsize=8.5, pad=2)
        tip_inset(ax, pr, c)
        fam, prm = pr.family, pr.p
        tangent_sh = abs(slope(pr, Ls * (1 - 1e-6), Ls, Rs)) < 1e-3
        if fam == "cone":
            thc = np.degrees(np.arctan(Rs / Ls))
            ax.add_patch(Arc((0, 0), 2.4, 2.4, theta1=0, theta2=thc, lw=0.6, color=INK))
            ax.text(1.28, 0.06, r"$\theta_c=\arctan(R/L)$", fontsize=7, color=INK, va="bottom")
            kink(ax, pr, c, True)
            bottom(ax, "straight generator, sharp tip; simplest mould")
        elif fam == "tangent_ogive":
            rho = (Rs**2 + Ls**2) / (2 * Rs)
            xa = 0.45 * Ls
            ya = radius(pr, xa, Ls, Rs)
            dvec = np.array([Ls - xa, (Rs - rho) - ya])
            dvec /= np.hypot(*dvec)
            t = (-1.22 - ya) / dvec[1]
            ax.annotate("", xy=(xa + t * dvec[0], ya + t * dvec[1]), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=c, ls=(0, (3, 2)), mutation_scale=6))
            ax.annotate("", xy=(Ls, -1.22), xytext=(Ls, -Rs),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=c, ls=(0, (3, 2)), mutation_scale=6))
            ax.text(xa + 0.5 * t * dvec[0] - 0.10, ya + 0.5 * t * dvec[1], r"$\rho$", fontsize=8, ha="right")
            tangent(ax)
            bottom(ax, r"arc of radius $\rho=(R^2+L^2)/2R$,", "centre in the base plane, below the axis")
        elif fam == "parabolic":
            (tangent(ax) if tangent_sh else kink(ax, pr, c, False))
            bottom(ax, "parabola with its vertex at the shoulder;" if prm["K"] == 1.0 else "parabolic series;",
                   "sharp tip with a finite angle")
        elif fam == "power":
            kink(ax, pr, c, False)
            if prm["n"] == 0.5:
                bottom(ax, r"blunt, rounded tip,", r"curvature radius $R^2/2L$ at the apex")
            else:
                bottom(ax, "pointed tip with a vertical tangent", "(zero tip radius)")
        elif fam == "haack":
            tangent(ax)
            if abs(prm["C"] - 1 / 3) < 1e-12:
                words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
                nw = words[len(order) - 1] if len(order) <= 12 else str(len(order))
                bottom(ax, r"least wave drag for given $L$ and volume;", f"the fullest of the {nw} shapes")
            elif prm["C"] == 0:
                bottom(ax, r"least wave drag for given $L$ and base area;", "pointed tip with a vertical tangent")
            else:
                bottom(ax, rf"Haack series with $C={prm['C']:g}$;", "pointed tip with a vertical tangent")
        else:
            (tangent(ax) if tangent_sh else kink(ax, pr, c, False))
            bottom(ax, _family_doc(pr))
    for k in range(len(order) + 1, len(axs)):
        axs[k].axis("off")
    fig.subplots_adjust(hspace=0.22, wspace=0.05)
    ctx.save(fig, "fig_schematics")


def _family_doc(pr: Profile) -> str:
    from ..geometry import FAMILIES
    return FAMILIES[pr.family].doc.split(":")[0]


# ---------------------------------------------------------------------------------- renders
def fig_renders(ctx: PlotContext) -> None:
    """3D renders of every profile at the design fineness and of the selected design."""
    from matplotlib.colors import LightSource, to_rgb
    from ..export import outline
    cfg, R = ctx.cfg, ctx.R
    p, f, rnr = ctx.study.design
    sh = cfg.nose.shoulder_length
    order = ctx.order()

    def render(ax, x, r, color, tip_x=None, n_az=90, zoom=1.9):
        th = np.linspace(0, 2 * np.pi, n_az)
        Xs = np.repeat(x[:, None], n_az, 1)
        Ys = r[:, None] * np.cos(th)
        Zs = r[:, None] * np.sin(th)
        ls = LightSource(azdeg=300, altdeg=35)
        base = np.array(to_rgb(color))
        rgb = np.ones(Xs.shape + (3,)) * (0.55 * base + 0.45)
        if tip_x is not None:
            rgb[Xs <= tip_x] = np.array(to_rgb("#9aa4ad"))
        shaded = ls.shade_rgb(rgb, Zs, fraction=0.9, blend_mode="soft")
        ax.plot_surface(Xs, Ys, Zs, facecolors=shaded, linewidth=0, antialiased=False, shade=False,
                        rstride=1, cstride=1)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((x.max() - x.min(), 2 * r.max(), 2 * r.max()), zoom=zoom)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-r.max(), r.max())
        ax.set_zlim(-r.max(), r.max())
        ax.view_init(elev=14, azim=-78)
        ax.set_axis_off()

    nrow = int(np.ceil((len(order) + 1) / 2))
    fig = plt.figure(figsize=(fs.WIDTH, 0.80 * fs.WIDTH * ((nrow + 1.5) / 5.5)))
    gs = fig.add_gridspec(nrow + 1, 2, height_ratios=[1] * nrow + [1.5], hspace=0.0, wspace=0.0,
                          left=0.0, right=1.0, top=0.97, bottom=0.0)
    L = 2 * R * f
    for k, n in enumerate(order):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection="3d")
        x, r = outline(ctx.p(n), L, R, sh, n=160)
        render(ax, x, r, ctx.sty(n)[0])
        ax.text2D(0.02, 0.80, ctx.label(n), transform=ax.transAxes, fontsize=8, color=INK)
    ax = fig.add_subplot(gs[len(order) // 2, 1] if len(order) % 2 else gs[nrow - 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, rf"all at $L/D = {f:g}$ ($L = {1e3 * L:.0f}$ mm), $D = {1e3 * 2 * R:.0f}$ mm," "\n"
            rf"with a {1e3 * sh:.0f} mm shoulder; same scale", ha="center", va="center", fontsize=7.5, color=INK)
    ax = fig.add_subplot(gs[nrow, :], projection="3d")
    rn = rnr * R
    x, r = outline(ctx.p(p), L, R, sh, rn=rn, n=300)
    mats = ctx.study.materials()
    render(ax, x, r, ctx.sty(p)[0], tip_x=1e3 * cfg.heating.tip_length)
    ax.set_title(tex(rf"\textbf{{Recommended:}} {ctx.label(p)}, $L/D={f:g}$, {1e3 * rn:g} mm "
                     rf"{mats[cfg.heating.tip_material].label} tip insert (grey)"), fontsize=8.5, pad=-2, color=INK)
    for fmt in cfg.output.formats:
        if fmt == "png":
            fig.savefig(ctx.outdir / "fig_renders.png", dpi=cfg.output.png_dpi)
        else:
            fig.savefig(ctx.outdir / f"fig_renders.{fmt}", dpi=300, bbox_inches=None)
        ctx.saved.append(ctx.outdir / f"fig_renders.{fmt}")
    plt.close(fig)


PLOTS: dict[str, Callable[[PlotContext], None]] = {
    "profiles": fig_profiles, "schematics": fig_schematics, "validation": fig_validation, "cdw": fig_cdw,
    "cp": fig_cp, "trajectory": fig_trajectory, "breakdown": fig_breakdown, "trade": fig_trade,
    "robustness": fig_robustness, "bluntness": fig_bluntness, "thermal": fig_thermal, "final": fig_final,
    "gallery": fig_gallery, "fields": fig_fields, "grid": fig_grid, "renders": fig_renders,
}
"""Registry of plots by name (see output.plots)."""

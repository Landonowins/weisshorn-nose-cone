"""Figures, data tables and numbers of 2027_C_FD_REPORT-5_NOSE-CONE-720MM (aerodynamic inputs for ST).

Reads the Fluent study of the two 720 mm noses (CFD/study720: wall data, C_D table, h(x)) and the
calibrated Python model (configs/weisshorn_720.yaml, outputs of studies/final_design.py), and writes

* figures/fig_*.pdf: the report figures made here (the Fluent figures come from CFD/study720/figures);
* data/*.csv: the tables delivered to ST (wall pressure and heating along x, histories in time);
* sections/numbers.tex: every number quoted in the text and tables, as \\val{key}.

Load cases (nominal trajectory and hot-day envelope):
    LC1 burnout = maximum Mach and dynamic pressure, LC2 transonic M 0.95 on the ascent, LC3 apogee.
Sealed nose: inside pressure p_in = p_pad T_in / T_pad, between T_in = T_pad (lower bound) and the
mean inner-face temperature of the shell (upper bound). Net load dp = p_wall - p_in, > 0 inwards.

Usage:
    python studies/final_design.py configs/weisshorn_720.yaml   # first
    python report_720/figures.py configs/weisshorn_720.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

import final_design as fd  # noqa: E402
import total_length as tl  # noqa: E402
from nosecone import heating as ht  # noqa: E402
from nosecone.flight import skin_friction  # noqa: E402
from nosecone.geometry import volume_and_area  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
CFD = ROOT / "CFD" / "study720"
GAMMA_F, R_F = 1.399, 287.04            # Fluent air (c_p 1006.43 J/(kg K), M 28.966)
T_WALL = 300.0                          # wall of the isothermal Fluent runs [K]
RUNS = {"M0.95": 0.95, "M1.20": 1.20, "adiab": None}      # adiabatic Fluent runs; None = burnout
SMOOTH = (3e-3, 15e-3, 2e-3)            # tip sawtooth: x range and moving-average window [m]
CASES = {"nominal": False, "hot": True}
SHAPE_COL = {"A": fs.COLORS["lvhaack"], "B": fs.COLORS["vonkarman"]}
CASE_LS = {"nominal": "-", "hot": (0, (5, 2))}
LIM_DP490, LIM_REQ10 = 120.0, 340.0 - 273.15


# ------------------------------------------------------------------------------ inputs
def wall(run: str) -> pd.DataFrame:
    """Fluent wall export of a run (wall_nose + wall_coupler), sorted along x, SI units."""
    d = pd.read_csv(CFD / "wall_data" / f"{run}_wall.csv", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"x-coordinate": "x", "y-coordinate": "r", "wall-shear": "tau", "y-plus": "yplus",
                          "heat-flux": "q", "temperature": "T", "pressure-coefficient": "Cp", "pressure": "p"})
    return d.sort_values("x").reset_index(drop=True)


def smooth_tip(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Moving average over SMOOTH[2] for SMOOTH[0] <= x <= SMOOTH[1] (sawtooth at the sphere-of-influence edge)."""
    out = v.copy()
    a, b, w = SMOOTH
    for i in np.where((x >= a) & (x <= b))[0]:
        out[i] = v[np.abs(x - x[i]) <= w / 2].mean()
    return out


def free_stream() -> pd.DataFrame:
    """Free stream of the Fluent runs (design A's trajectory points, used for both designs)."""
    f = pd.read_csv(CFD / "fluent_points_python.csv")
    f = f[f.design == "A"].set_index("point")
    f["rho"] = f.p_Pa / (R_F * f.T_K)
    f["V"] = f.mach * np.sqrt(GAMMA_F * R_F * f.T_K)
    f["q"] = 0.5 * f.rho * f.V**2
    return f


def ring_area_x(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Projected (axial) area 2 pi r dr of each face, from face centres (edges at the mid-points)."""
    re = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [r[-1]]])
    return np.pi * (re[1:] ** 2 - re[:-1] ** 2)


def ring_area_s(x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wetted area 2 pi r ds and axial length dx of each face."""
    xe = np.concatenate([[0.0], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    re = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [r[-1]]])
    ds = np.hypot(np.diff(xe), np.diff(re))
    return 2 * np.pi * r * ds, np.diff(xe)


# ------------------------------------------------------------------------------ model
class Design:
    """One design: geometry, trajectories, Fluent wall data and the derived load histories."""

    def __init__(self, mdl: tl.Model, study: str, summary: dict, fsm: pd.DataFrame):
        self.mdl, self.study = mdl, study
        eq = summary["equations"][study]
        self.eq, self.shape, self.L, self.c = eq, eq["shape"], eq["L"], eq["c"]
        self.xS = eq["L_phys"]
        self.loads = summary["loads"][study]
        self.walls = {k: wall(f"{study}_{k}") for k in RUNS}
        self.cold = wall(f"{study}_T300")
        self.hx = pd.read_csv(CFD / f"{study}_h_x.csv")
        self.x = self.walls["adiab"].x.values
        self.r = self.walls["adiab"].r.values
        self.M_runs = np.array([fsm.loc["M0.95", "mach"], fsm.loc["M1.20", "mach"], fsm.loc["burnout", "mach"]])
        self.cp = np.array([self.walls[k].Cp.values for k in RUNS])      # (3, n) at M_runs
        self.fsm = fsm
        self.tr = {c: mdl.trajectory(self.shape, self.L, self.c, envelope=e, detail=True) for c, e in CASES.items()}
        self.hist = {c: pd.read_csv(Path(mdl.cfg.project.output_dir) / "final" / f"heating_history_{study}_"
                                    f"{'nominal' if c == 'nominal' else 'envelope'}.csv") for c in CASES}
        jf = pd.read_csv(Path(mdl.cfg.project.output_dir) / "final" / f"joint_force_{study}.csv")
        self.Dn = {c: jf[jf.case == ("nominal" if c == "nominal" else "envelope")] for c in CASES}
        self.T_in = {c: self.inner_air(c) for c in CASES}

    def cp_at(self, M: float) -> np.ndarray:
        """Cp(x) at Mach M: linear in M between the Fluent runs, held outside them."""
        M = float(np.clip(M, self.M_runs[0], self.M_runs[-1]))
        k = int(np.clip(np.searchsorted(self.M_runs, M) - 1, 0, 1))
        w = (M - self.M_runs[k]) / (self.M_runs[k + 1] - self.M_runs[k])
        return (1 - w) * self.cp[k] + w * self.cp[k + 1]

    def inner_air(self, case: str) -> np.ndarray:
        """Upper bound of the inside air temperature: mean inner-face temperature at 50 mm, mid-length and
        shoulder, never below T_pad (the trapped air lags the shell when it cools in the coast)."""
        h = self.mdl.cfg.heating
        mats = self.mdl.st.materials()
        tr = self.tr[case]
        T = [ht.wall_response(tr, mats[h.shell_material], h.shell_thickness, x, n=h.wall_nodes,
                              **ht.calibration(h))["T_in"] for x in (0.05, 0.5 * self.xS, self.xS)]
        return np.maximum(tr["T"][0], np.concatenate([[tr["T"][0]], np.mean(T, axis=0)]))

    def p_in(self, case: str, upper: bool) -> np.ndarray:
        tr = self.tr[case]
        return tr["p"][0] * (self.T_in[case] if upper else np.full_like(tr["t"], tr["T"][0])) / tr["T"][0]

    def p_wall(self, case: str, i: int) -> np.ndarray:
        tr = self.tr[case]
        return tr["p"][i] + self.cp_at(tr["M"][i]) * tr["q"][i]

    def lc_index(self, case: str) -> dict:
        tr = self.tr[case]
        i_max = int(np.argmax(tr["M"]))
        i_95 = int(np.argmax(tr["M"][:i_max + 1] >= 0.95))
        return {"LC1": i_max, "LC2": i_95, "LC3": len(tr["t"]) - 1}

    def stations(self) -> dict:
        """Wall faces of the pressure histories: stagnation point, 50 mm, mid-length, shoulder."""
        pick = lambda xx: int(np.argmin(np.abs(self.x - xx)))  # noqa: E731
        return {"tip": 0, "50 mm": pick(0.05), "mid": pick(0.5 * self.xS), "shoulder": pick(self.xS - 1e-3)}

    def dp_history(self, case: str) -> pd.DataFrame:
        tr = self.tr[case]
        st = self.stations()
        pin_lo, pin_hi = self.p_in(case, False), self.p_in(case, True)
        d = dict(t=tr["t"], M=tr["M"], q=tr["q"], p_inf=tr["p"], p_in_low=pin_lo, p_in_high=pin_hi)
        cpm = np.array([self.cp_at(m)[list(st.values())] for m in tr["M"]])
        for k, (name, _) in enumerate(st.items()):
            pw = tr["p"] + cpm[:, k] * tr["q"]
            d[f"p_wall_{name}"] = pw
            d[f"dp_{name}_low"] = pw - pin_lo      # p_in at its lower bound: largest inward load
            d[f"dp_{name}_high"] = pw - pin_hi     # upper bound: largest outward load
        return pd.DataFrame(d)

    def fluent_forces(self) -> pd.DataFrame:
        """Axial pressure and viscous force on wall_nose and wall_coupler from the wall data [N]."""
        rows = []
        for k, M in zip(RUNS, self.M_runs):
            w = self.walls[k]
            pt = "burnout" if RUNS[k] is None else k
            p_inf, q = self.fsm.loc[pt, "p_Pa"], self.fsm.loc[pt, "q"]
            Ax = ring_area_x(w.x.values, w.r.values)
            _, dx = ring_area_s(w.x.values, w.r.values)
            nose = w.x.values <= self.xS
            Fp = (w.p.values - p_inf) * Ax
            Fv = w.tau.values * 2 * np.pi * w.r.values * dx
            rows.append(dict(run=k, M=M, q=q, Fp_nose=Fp[nose].sum(), Fv_nose=Fv[nose].sum(),
                             Fv_coupler=Fv[~nose].sum(), F_total=Fp.sum() + Fv.sum()))
        return pd.DataFrame(rows)

    def normal_load(self) -> pd.DataFrame:
        """Slender body: dN/dx per degree at LC1 (nominal and hot), and the nose CP."""
        x = np.linspace(0.0, self.xS, 1401)
        r = fd.r_of_x(self.eq, x)
        dS = np.gradient(np.pi * r**2, x)
        d = dict(x_mm=1e3 * x, r_mm=1e3 * r)
        for c in CASES:
            q = self.tr[c]["q"][self.lc_index(c)["LC1"]]
            d[f"dNdx_per_deg_{c}_N_per_m"] = 2 * q * np.radians(1.0) * dS
        return pd.DataFrame(d)

    def x_cp(self) -> float:
        V, _ = volume_and_area(self.mdl.pr(self.shape), self.L, self.mdl.R, self.mdl.rn)
        return self.xS - V / (np.pi * self.mdl.R**2)


# ------------------------------------------------------------------------------ figures
def fig_flight(des: dict, fsm: pd.DataFrame, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.36 * fs.WIDTH))
    for s, d in des.items():
        for c in CASES:
            tr = d.tr[c]
            a.plot(tr["t"], tr["M"], color=SHAPE_COL[s], ls=CASE_LS[c], lw=1.0)
            b.plot(tr["t"], 1e-3 * tr["q"], color=SHAPE_COL[s], ls=CASE_LS[c], lw=1.0)
    for pt, row in fsm.iterrows():
        a.plot(row.t_s, row.mach, "o", mfc="white", mec=fs.INK, ms=4, zorder=5)
        b.plot(row.t_s, 1e-3 * row.q, "o", mfc="white", mec=fs.INK, ms=4, zorder=5)
    a.set(xlim=(0, 45), ylim=(0, 1.7), xlabel=r"$t$ [s]", ylabel=r"Mach number $M$")
    b.set(xlim=(0, 45), ylim=(0, 140), xlabel=r"$t$ [s]", ylabel=r"dynamic pressure $q$ [kPa]")
    b.legend(handles=[Line2D([], [], color=SHAPE_COL[s], label=f"design {s}") for s in des]
             + [Line2D([], [], color=fs.INK2, ls=CASE_LS[c], label=("nominal" if c == "nominal" else "hot day"))
                for c in CASES]
             + [Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK, label="Fluent runs")],
             loc="upper right", fontsize=7)
    fs.panel(a, "a"), fs.panel(b, "b")
    fs.save(fig, "fig_flight", out)


def fig_dp_x(des: dict, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    lcs = {"LC1": ("burnout", "#d03b3b"), "LC2": ("$M=0.95$", "#2a78d6"), "LC3": ("apogee", "#1baf7a")}
    for a, (s, d) in zip(axs, des.items()):
        for c in CASES:
            idx = d.lc_index(c)
            for lc, (lab, col) in lcs.items():
                i = idx[lc]
                pw = smooth_tip(d.x, d.p_wall(c, i))
                lo, hi = d.p_in(c, False)[i], d.p_in(c, True)[i]
                a.plot(1e3 * d.x, 1e-3 * (pw - lo), color=col, ls=CASE_LS[c], lw=1.0)
                if lc == "LC3":
                    a.plot(1e3 * d.x, 1e-3 * (pw - hi), color=col, ls=CASE_LS[c], lw=0.6)
        a.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
        a.axvline(1e3 * d.xS, color=fs.INK2, lw=0.5, ls=(0, (1, 2)))
        a.set(xlim=(0, 720), xlabel=r"$x$ from the tip [mm]", title=f"design {s}")
        a.set_xscale("symlog", linthresh=20, linscale=0.6)
        a.set_xticks([0, 10, 20, 50, 100, 200, 400, 720])
        a.set_xticklabels(["0", "10", "20", "50", "100", "200", "400", "720"])
    axs[0].set(ylabel=r"net pressure $\Delta p = p_w - p_{in}$ [kPa]", ylim=(-90, 170))
    axs[1].legend(handles=[Line2D([], [], color=col, label=f"{lc}, {lab}") for lc, (lab, col) in lcs.items()]
                  + [Line2D([], [], color=fs.INK2, ls=CASE_LS[c], label=("nominal" if c == "nominal" else "hot day"))
                     for c in CASES], loc="upper right", fontsize=7)
    fs.panel(axs[0], "a", y=0.9), fs.panel(axs[1], "b", y=0.9)
    fs.save(fig, "fig_dp_x", out)


def fig_dp_t(des: dict, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    cols = {"tip": "#d03b3b", "50 mm": "#eb6834", "mid": "#2a78d6", "shoulder": "#1baf7a"}
    for a, (s, d) in zip(axs, des.items()):
        for c in CASES:
            h = d.dp_history(c)
            for name, col in cols.items():
                a.plot(h.t, 1e-3 * h[f"dp_{name}_low"], color=col, ls=CASE_LS[c], lw=1.0)
                a.plot(h.t, 1e-3 * h[f"dp_{name}_high"], color=col, ls=CASE_LS[c], lw=0.5)
        a.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
        a.set(xlim=(0, 45), xlabel=r"$t$ [s]", title=f"design {s}")
    axs[0].set(ylabel=r"$\Delta p = p_w - p_{in}$ [kPa]", ylim=(-90, 170))
    axs[1].legend(handles=[Line2D([], [], color=col, label=n) for n, col in cols.items()]
                  + [Line2D([], [], color=fs.INK2, lw=1.0, label=r"$p_{in}$ at $T_{pad}$"),
                     Line2D([], [], color=fs.INK2, lw=0.5, label=r"$p_{in}$, heated air")],
                  loc="upper right", fontsize=7)
    fs.panel(axs[0], "a", y=0.9), fs.panel(axs[1], "b", y=0.9)
    fs.save(fig, "fig_dp_t", out)


def fig_cd(des: dict, mdl650: tl.Model, out: Path) -> None:
    fig, a = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.42 * fs.WIDTH))
    fl = pd.read_csv(CFD / "fluent_cd_720.csv")
    for s, d in des.items():
        tr = d.tr["nominal"]
        up = slice(0, int(np.argmax(tr["M"])) + 1)
        M = tr["M"][up]
        Re = tr["rho"][up] * tr["V"][up] * d.xS / tr["mu"][up]
        _, S = volume_and_area(d.mdl.pr(d.shape), d.L, d.mdl.R, d.mdl.rn)
        cdf = skin_friction(Re, M, d.xS, 0.0) * S / (np.pi * d.mdl.R**2)
        new = d.mdl.vehicle(d.shape, d.L, d.c).cdw(M) + cdf
        old = mdl650.vehicle(d.shape, d.L, d.c).cdw(M) + cdf
        m = M > 0.3
        a.plot(M[m], new[m], color=SHAPE_COL[s], lw=1.2, label=f"{s}: model, rise from $M=0.94$")
        a.plot(M[m], old[m], color=SHAPE_COL[s], lw=0.7, ls=(0, (2, 2)), label=f"{s}: model, rise from $M=0.8$")
        f = fl[fl.design == s].sort_values("mach")
        a.plot(f.mach, f.cd, "o", color=SHAPE_COL[s], mfc="white", ms=5, label=f"{s}: Fluent")
    a.set(xlim=(0.3, 1.5), ylim=(0, 0.11), xlabel=r"Mach number $M$", ylabel=r"nose $C_D$ (smooth wall)")
    a.legend(loc="lower right", fontsize=6.3, ncol=2, columnspacing=1.0)
    fs.save(fig, "fig_cd_mach", out)


def fig_axial(des: dict, ff: dict, out: Path) -> None:
    fig, a = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.40 * fs.WIDTH))
    for s, d in des.items():
        for c in CASES:
            j = d.Dn[c]
            a.plot(j.t, j.D_n, color=SHAPE_COL[s], ls=CASE_LS[c], lw=1.0)
        fsm = d.fsm
        for _, row in ff[s].iterrows():
            pt = "burnout" if row.run == "adiab" else row.run
            a.plot(fsm.loc[pt, "t_s"], row.F_total, "o", color=SHAPE_COL[s], mfc="white", ms=5)
    a.set(xlim=(0, 30), ylim=(0, 350), xlabel=r"$t$ [s]", ylabel=r"nose axial force $D_n$ [N]")
    a.legend(handles=[Line2D([], [], color=SHAPE_COL[s], label=f"design {s}") for s in des]
             + [Line2D([], [], color=fs.INK2, ls=CASE_LS[c], label=("model, nominal" if c == "nominal" else "model, hot day"))
                for c in CASES]
             + [Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK2, label="Fluent (smooth wall)")],
             loc="upper right", fontsize=7)
    fs.save(fig, "fig_axial_force", out)


def fig_normal(des: dict, out: Path) -> None:
    fig, a = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.40 * fs.WIDTH))
    for s, d in des.items():
        nl = d.normal_load()
        for c in CASES:
            a.plot(nl.x_mm, nl[f"dNdx_per_deg_{c}_N_per_m"], color=SHAPE_COL[s], ls=CASE_LS[c], lw=1.0)
        a.axvline(1e3 * d.x_cp(), color=SHAPE_COL[s], lw=0.6, ls=(0, (1, 2)))
    a.set(xlim=(0, 720), ylim=(0, None), xlabel=r"$x$ from the tip [mm]",
          ylabel=r"$\mathrm{d}N/\mathrm{d}x$ per degree [N/m]")
    a.legend(handles=[Line2D([], [], color=SHAPE_COL[s], label=f"design {s}") for s in des]
             + [Line2D([], [], color=fs.INK2, ls=CASE_LS[c], label=("LC1 nominal" if c == "nominal" else "LC1 hot day"))
                for c in CASES]
             + [Line2D([], [], color=fs.INK2, lw=0.6, ls=(0, (1, 2)), label=r"$x_{CP}$")], loc="lower center",
             fontsize=7, ncol=3, columnspacing=1.0)
    fs.save(fig, "fig_normal_load", out)


def fig_h(des: dict, fsm: pd.DataFrame, out: Path) -> None:
    h = des["A"].mdl.cfg.heating
    M, p, T = fsm.loc["burnout", ["mach", "p_Pa", "T_K"]]
    V = M * np.sqrt(GAMMA_F * R_F * T)
    fig, a = plt.subplots(figsize=(0.62 * fs.WIDTH, 0.42 * fs.WIDTH))
    x = np.linspace(0.005, 0.72, 300)
    raw = np.array([ht.h_turbulent(M, p, T, V, xx, T_WALL)[0] for xx in x])
    for s, d in des.items():
        hx = d.hx
        a.plot(hx.x_mm, smooth_tip(1e-3 * hx.x_mm.values, hx.h_W_m2K.values), color=SHAPE_COL[s], lw=1.0,
               label=f"{s}: Fluent")
    a.plot(1e3 * x, raw, color=fs.INK2, lw=0.7, ls=(0, (2, 2)), label="model, uncalibrated")
    a.plot(1e3 * x, h.calibration_turbulent * raw, color=fs.INK, lw=1.0, ls=(0, (5, 2)),
           label=rf"model $\times\,k_t = {h.calibration_turbulent:g}$")
    hs = ht.h_stagnation(M, p, T, des["A"].mdl.rn)[0]
    a.plot(0.5, hs, "s", color=fs.INK2, mfc="white", ms=4, label="Fay--Riddell, uncalibrated")
    a.plot(0.5, h.calibration_stagnation * hs, "s", color=fs.INK, ms=4,
           label=rf"Fay--Riddell $\times\,k_s = {h.calibration_stagnation:g}$")
    a.set_xscale("log")
    a.set(xlim=(0.4, 800), ylim=(0, 2200), xlabel=r"$x$ from the tip [mm]",
          ylabel=r"$h = q_w/(T_{aw} - 300\,\mathrm{K})$ [W/(m$^2$K)]")
    a.legend(loc="upper right", fontsize=6.5)
    fs.save(fig, "fig_h_calibration", out)


def fig_temps(des: dict, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    cols = {"T_tip_insert": ("Al tip insert", "#d03b3b"), "T_shell_out_50mm": ("shell outer face, 50 mm", "#eb6834"),
            "T_shell_in_50mm": ("shell inner face, 50 mm", "#2a78d6")}
    for a, (s, d) in zip(axs, des.items()):
        for c in CASES:
            hh = d.hist[c]
            for k, (_, col) in cols.items():
                a.plot(hh.t, hh[k] - 273.15, color=col, ls=CASE_LS[c], lw=1.0)
            a.plot(d.tr[c]["t"], d.T_in[c] - 273.15, color="#1baf7a", ls=CASE_LS[c], lw=0.8)
        a.axhline(LIM_DP490, color=fs.INK, lw=0.6, ls=(0, (4, 2)))
        a.text(44, LIM_DP490 + 2, "DP490 120 $^\\circ$C", ha="right", fontsize=7)
        a.axhline(LIM_REQ10, color=fs.INK2, lw=0.6, ls=(0, (1, 2)))
        a.text(44, LIM_REQ10 + 2, "Req\\_10 (340 K)", ha="right", fontsize=7, color=fs.INK2)
        a.set(xlim=(0, 45), xlabel=r"$t$ [s]", title=f"design {s}")
    axs[0].set(ylim=(0, 130), ylabel=r"temperature [$^\circ$C]")
    fig.legend(handles=[Line2D([], [], color=col, label=lab) for lab, col in cols.values()]
               + [Line2D([], [], color="#1baf7a", lw=0.8, label="inside air (upper bound)")]
               + [Line2D([], [], color=fs.INK2, ls=CASE_LS[c], label=("nominal" if c == "nominal" else "hot day"))
                  for c in CASES], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7)
    fs.save(fig, "fig_temperatures", out)


# ------------------------------------------------------------------------------ numbers and data
def numbers(des: dict, ff: dict, fsm: pd.DataFrame) -> dict:
    v = {}
    put = lambda k, x, f="{:.0f}": v.__setitem__(k, f.format(x))  # noqa: E731
    h = des["A"].mdl.cfg.heating
    put("ks", h.calibration_stagnation, "{:g}"), put("kt", h.calibration_turbulent, "{:g}")
    put("ktip", h.calibration_tip, "{:g}")
    fl = pd.read_csv(CFD / "fluent_cd_720.csv")
    for s, d in des.items():
        put(f"{s}-xS", 1e3 * d.xS, "{:.1f}"), put(f"{s}-c", 1e3 * d.c)
        put(f"{s}-xcp", 1e3 * d.x_cp())
        put(f"{s}-apogee", d.tr["nominal"]["apogee"])
        for _, row in fl[fl.design == s].iterrows():
            k = {"0.95": "M095", "1.20": "M120", "1.46548": "M147"}[f"{row.mach:.2f}" if row.mach < 1.4 else "1.46548"]
            put(f"{s}-cd-{k}", row.cd, "{:.4f}"), put(f"{s}-cdp-{k}", row.cd_p, "{:.4f}")
            put(f"{s}-cdv-{k}", row.cd_v, "{:.4f}")
        for _, row in ff[s].iterrows():
            k = {"M0.95": "M095", "M1.20": "M120", "adiab": "M147"}[row.run]
            put(f"{s}-Fp-{k}", row.Fp_nose, "{:.1f}"), put(f"{s}-Fv-{k}", row.Fv_nose, "{:.1f}")
            put(f"{s}-Fc-{k}", row.Fv_coupler, "{:.1f}"), put(f"{s}-F-{k}", row.F_total, "{:.1f}")
        put(f"{s}-L", 1e3 * d.L, "{:.1f}"), put(f"{s}-delta", 1e3 * d.eq["delta"], "{:.1f}")
        put(f"{s}-fineness", d.L / (2 * d.mdl.R), "{:.2f}")
        put(f"{s}-apogee-hot", d.tr["hot"]["apogee"])
        hx = d.hx
        put(f"{s}-qw0", 1e-3 * hx.q_w_300K_W_m2.iloc[0], "{:.1f}"), put(f"{s}-h0", hx.h_W_m2K.iloc[0])
        put(f"{s}-Taw0", hx.T_aw_K.iloc[0], "{:.1f}")
        sh = hx.iloc[(hx.x_mm - 1e3 * d.xS).abs().argmin()]
        put(f"{s}-hS", sh.h_W_m2K)
        for c in CASES:
            tr, idx = d.tr[c], d.lc_index(c)
            dpt = d.dp_history(c)
            for lc, i in idx.items():
                pre = f"{s}-{c}-{lc}"
                put(f"{pre}-t", tr["t"][i], "{:.1f}"), put(f"{pre}-M", tr["M"][i], "{:.2f}")
                put(f"{pre}-q", 1e-3 * tr["q"][i], "{:.1f}"), put(f"{pre}-pinf", 1e-3 * tr["p"][i], "{:.1f}")
                pw = smooth_tip(d.x, d.p_wall(c, i))
                lo, hi = d.p_in(c, False)[i], d.p_in(c, True)[i]
                put(f"{pre}-dpstag", 1e-3 * (pw[0] - lo), "{:.0f}")
                put(f"{pre}-dpmax", 1e-3 * (pw - lo).max(), "{:.0f}")
                put(f"{pre}-dpmax2", 1e-3 * (pw - lo)[d.x > 2e-3].max(), "{:.0f}")   # beyond the tip cap
                put(f"{pre}-dpmin", 1e-3 * (pw - hi).min(), "{:.0f}")
                put(f"{pre}-Dn", d.Dn[c].D_n.values[i])
                put(f"{pre}-Nperdeg", 2 * tr["q"][i] * np.pi * d.mdl.R**2 * np.radians(1.0))
                put(f"{pre}-pin", 1e-3 * hi, "{:.0f}")
                Ax = ring_area_x(d.x, d.r)              # net pressure force on the joint, > 0 compression
                put(f"{pre}-Fplo", np.sum((pw - lo) * Ax))
                put(f"{pre}-Fphi", np.sum((pw - hi) * Ax))
            p0 = ht.stagnation_state(tr["M"][idx["LC1"]], tr["p"][idx["LC1"]], tr["T"][idx["LC1"]])[0]
            put(f"{s}-{c}-p0check", 100 * ((d.p_wall(c, idx["LC1"])[0]) / p0 - 1), "{:+.1f}")
            put(f"{s}-{c}-dpminall", 1e-3 * min(dpt[[k for k in dpt if k.endswith("_high")]].min()), "{:.0f}")
            put(f"{s}-{c}-dpmaxall", 1e-3 * max(dpt[[k for k in dpt if k.endswith("_low")]].max()), "{:.0f}")
            hh = d.hist[c]
            put(f"{s}-{c}-Ttip", hh.T_tip_insert.max() - 273.15)
            put(f"{s}-{c}-Tout50", hh.T_shell_out_50mm.max() - 273.15)
            put(f"{s}-{c}-Tin50", hh.T_shell_in_50mm.max() - 273.15)
            put(f"{s}-{c}-Tair", d.T_in[c].max() - 273.15)
            put(f"{s}-{c}-T0", hh.T0.max() - 273.15)
            j = d.Dn[c]
            put(f"{s}-{c}-Dnmax", j.D_n.max()), put(f"{s}-{c}-tDnmax", j.t.values[int(j.D_n.values.argmax())], "{:.1f}")
    for k in ("M095", "M120", "M147"):
        put(f"BvsA-{k}", 100 * (float(v[f"B-cd-{k}"]) / float(v[f"A-cd-{k}"]) - 1), "{:+.0f}")
        put(f"BlessA-{k}", 100 * (1 - float(v[f"B-cd-{k}"]) / float(v[f"A-cd-{k}"])), "{:.0f}")
    for pt, row in fsm.iterrows():                      # free stream of the Fluent runs
        k = {"M0.95": "M095", "M1.20": "M120", "burnout": "M147"}[pt]
        put(f"pt-{k}-t", row.t_s, "{:.2f}"), put(f"pt-{k}-z", row.z_m), put(f"pt-{k}-M", row.mach, "{:.3f}")
        put(f"pt-{k}-p", row.p_Pa), put(f"pt-{k}-T", row.T_K, "{:.2f}"), put(f"pt-{k}-q", 1e-3 * row.q, "{:.1f}")
        put(f"pt-{k}-Re", row.rho * row.V / float(ht.sutherland(row.T_K)) * 1e-6, "{:.1f}")
    allp = pd.read_csv(CFD / "fluent_points_python.csv").set_index(["design", "point"])
    put("B-own-burnout-M", allp.loc[("B", "burnout"), "mach"], "{:.3f}")
    put("pad-p", 1e-3 * des["A"].tr["nominal"]["p"][0], "{:.1f}")
    put("pad-T", des["A"].tr["nominal"]["T"][0], "{:.2f}"), put("pad-Thot", des["A"].tr["hot"]["T"][0], "{:.2f}")
    return v


def write_data(des: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s, d in des.items():
        w = pd.DataFrame(dict(x_mm=1e3 * d.x, r_mm=1e3 * d.r))
        for k in RUNS:
            w[f"Cp_{'M147' if k == 'adiab' else k.replace('.', '')}"] = smooth_tip(d.x, d.walls[k].Cp.values)
        for c in CASES:
            for lc, i in d.lc_index(c).items():
                pw = smooth_tip(d.x, d.p_wall(c, i))
                w[f"p_wall_{lc}_{c}_kPa"] = 1e-3 * pw
                w[f"dp_{lc}_{c}_kPa"] = 1e-3 * (pw - (d.p_in(c, True)[i] if lc == "LC3" else d.p_in(c, False)[i]))
        hx = d.hx.set_index(d.hx.index)
        w["T_aw_LC1_K"] = d.hx.T_aw_K.values if len(d.hx) == len(w) else np.interp(w.x_mm, d.hx.x_mm, d.hx.T_aw_K)
        w["q_w_300K_LC1_kW_m2"] = 1e-3 * smooth_tip(d.x, np.interp(w.x_mm, d.hx.x_mm, d.hx.q_w_300K_W_m2))
        w["h_LC1_W_m2K"] = smooth_tip(d.x, np.interp(w.x_mm, d.hx.x_mm, d.hx.h_W_m2K))
        w.round(5).to_csv(out / f"{s}_wall_x.csv", index=False)
        hist = []
        for c in CASES:
            dp = d.dp_history(c)
            dp.insert(0, "case", c)
            dp["D_n_N"] = d.Dn[c].D_n.values
            hh = d.hist[c]
            for k in ("T_tip_insert", "T_shell_out_50mm", "T_shell_in_50mm"):
                dp[k + "_C"] = hh[k].values - 273.15
            dp["T_air_upper_C"] = d.T_in[c] - 273.15
            hist.append(dp)
        pd.concat(hist).round(4).to_csv(out / f"{s}_history_t.csv", index=False)
        d.normal_load().round(4).to_csv(out / f"{s}_normal_load_x.csv", index=False)


def write_numbers(v: dict, path: Path) -> None:
    lines = ["% generated by report_720/figures.py, do not edit",
             r"\providecommand{\val}[1]{\csname v@#1\endcsname}"]
    lines += [rf"\expandafter\def\csname v@{k}\endcsname{{{x}}}" for k, x in sorted(v.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    a = ap.parse_args()
    fs.setup()
    mdl = tl.Model(Path(a.config))
    mdl650 = tl.Model(ROOT / "configs" / "weisshorn_650.yaml")
    summary = json.loads((Path(mdl.cfg.project.output_dir) / "final" / "summary.json").read_text(encoding="utf-8"))
    fsm = free_stream()
    des = {s: Design(mdl, s, summary, fsm) for s in ("A", "B")}
    ff = {s: d.fluent_forces() for s, d in des.items()}
    figs = HERE / "figures"
    fig_flight(des, fsm, figs)
    fig_dp_x(des, figs)
    fig_dp_t(des, figs)
    fig_cd(des, mdl650, figs)
    fig_axial(des, ff, figs)
    fig_normal(des, figs)
    fig_h(des, fsm, figs)
    fig_temps(des, figs)
    write_data(des, HERE / "data")
    v = numbers(des, ff, fsm)
    write_numbers(v, HERE / "sections" / "numbers.tex")
    fl = pd.read_csv(CFD / "fluent_cd_720.csv")
    for s in des:                                       # wall-data integration against Fluent's report
        for _, row in ff[s].iterrows():
            r = fl[(fl.design == s) & (fl.run == f"{s}_{row.run}")].iloc[0]
            print(f"{s} {row.run:6s}: F_p {row.Fp_nose:7.1f} N (Fluent {r.cd_p * row.q * np.pi * 0.09**2:7.1f}), "
                  f"F_v {row.Fv_nose:6.1f} N (Fluent {r.cd_v * row.q * np.pi * 0.09**2:6.1f}), coupler {row.Fv_coupler:4.1f} N")
    for k in sorted(v):
        print(f"{k:28s} {v[k]}")


if __name__ == "__main__":
    main()

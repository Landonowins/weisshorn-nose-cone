"""Nose cone at a fixed total length, coupler included (2027_C_SE_ST_Nosecone/Payload-Bay_Req_03).

The coupler is a full-diameter cylinder behind the curved part; the coupler ring is bonded inside
it and carries the joint (separation plane) to the recovery (RE) bay. With the total length fixed,
every millimetre of coupler is taken from the curved part. For each profile and coupler length c
(curved length L = total - c):

* apogee. The vehicle below the nose joint is the same for every c; the coupler adds wetted area
  (skin friction) and shell mass. The 9 km reference is the LVS vehicle (Von Karman, 720 mm
  curved nose) as in report 1.
* GRV fit. The 3U payload (guided recovery vehicle, PL_GRV_REQ_02, 10 x 10 x 30 cm) needs an
  outer radius r >= r_req. Placed as far forward as possible, it fills the last l_in of the curved
  part, then the coupler, and reaches p = l_GRV - l_in - c beyond the joint (p <= 0: inside).

Two studies:

* A, 3U in the nose (Req_13; SE: if it does not fit, the cylinder is lengthened): shortest
  coupler >= C_MIN with p <= 0.
* B, 3U not in the nose: shortest coupler, C_MIN.

For each study the script gives the design point of each profile at 650 mm, the apogee against
the allowed total length (for the interface meeting), the loads at the nose joint, the heating,
the static-margin change and the CAD (STEP, STL, profile CSV) for the Fluent verification, and
writes the figures and LaTeX tables of report 2.

Usage:
    python studies/total_length.py configs/weisshorn_650.yaml [--report report_650] [--replot]
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
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nosecone import config as cfgmod  # noqa: E402
from nosecone import constants as NC  # noqa: E402
from nosecone import export  # noqa: E402
from nosecone import heating as ht  # noqa: E402
from nosecone.design import Study  # noqa: E402
from nosecone.flight import simulate, skin_friction  # noqa: E402
from nosecone.geometry import length_with_radius, surface, volume_and_area  # noqa: E402
from nosecone.pipeline import load_databases  # noqa: E402
from nosecone.stability import fin_cna_xcp  # noqa: E402

# ------------------------------------------------------------------------------- study inputs
TOTAL = 0.650              # Nosecone/Payload-Bay Req_03: total length, coupler included [m]
PROFILES = ["vonkarman", "lvhaack", "power050"]
PREF = ["vonkarman", "lvhaack", "power050"]   # tie-break: tangent base and pointed tip first
C_MIN = 0.020              # shortest coupler: bonded overlap of the coupler ring (to be confirmed by ST) [m]
C_SWEEP = np.round(np.arange(0.020, 0.2001, 0.005), 4)
T_SWEEP = np.round(np.arange(0.600, 0.8501, 0.010), 3)    # allowed total length, for the interface meeting [m]
T_TABLE = [650.0, 700.0, 720.0, 750.0, 800.0, 850.0]      # rows of the length table [mm]
LV_MAX = 5.4               # SE_REQ_04: launch vehicle length [m]
RULES = {                  # study -> largest GRV protrusion beyond the joint [m]
    "inside": 0.0,         # A: 3U entirely in the nose (Req_13)
    "free": math.inf,      # B: 3U not in the nose: shortest coupler
}
GRV_MASS = 3.99            # PL_GRV_REQ_03, upper bound [kg]
STRUCT_MAX = 1.400         # Req_04: nose with payload structure, without payload [kg]
EXTRA_STRUCT = 0.095       # reinforcement plies 35 g, tip ring 20 g, adhesive 25 g, paint 15 g (report 1) [kg]
CANSAT_D, CANSAT_H = 0.066, 0.115                       # CanSat can [m]
CANSAT_MASS = 0.350        # upper limit of the ESA CanSat format (66 x 115 mm can); not set by the ERT sheet [kg]
G_REQ07 = 20.0             # Req_07 rationale: 20 g deceleration on the nose mass, FoS 2
CANSAT_R = CANSAT_D / 2 + 0.003 + 0.002 + 0.0012          # + clearance + guide + shell, as for the GRV [m]
FOS = 2.0                  # ERT factor of safety of the structural requirements (Req_07)
REQ06_N, REQ12_NM = 300.0, 200.0                          # Req_06 compression, Req_12 bending (ultimate)
MIN_FINENESS = 2.5         # lowest L/D of the Euler database
TIE = 10.0                 # apogee differences below this are within the model uncertainty [m]
LABELS = {"vonkarman": "Von K\\'arm\\'an", "lvhaack": "LV-Haack", "power050": "Power $\\tfrac12$"}
PKEY = {"vonkarman": "VK", "lvhaack": "LVH", "power050": "PH"}
SKEY = {"inside": "In", "free": "Out"}                    # macro prefix of each study
STUDY = {"inside": "A", "free": "B"}
RULE_LABELS = {"inside": "A: 3U in the nose", "free": "B: 3U not in the nose"}


# --------------------------------------------------------------------------------- geometry
class Model:
    """Study objects shared by all evaluations."""

    def __init__(self, cfg_path: Path):
        self.cfg = cfgmod.load(cfg_path)
        self.st = Study(self.cfg, load_databases(self.cfg))
        self.R = self.st.R
        self.rn = self.cfg.nose.design_tip_radius_ratio * self.R
        self.l_grv = self.cfg.payload.length + self.cfg.payload.axial_clearance
        v = self.cfg.vehicle
        self.rest = v.length - v.reference_nose_length       # vehicle behind the nose joint [m]

    def pr(self, shape: str):
        return self.st.profiles[shape]

    # ---------------------------------------------------------------- payload fit
    def fit(self, shape: str, L: float, c: float, l_grv: float | None = None) -> dict:
        """GRV and CanSat positions [m, from the apex] for curved length L and coupler c."""
        p, pr = self.cfg.payload, self.pr(shape)
        lg = self.l_grv if l_grv is None else l_grv
        l_in = length_with_radius(pr, L, self.R, p.required_radius)
        x_grv = L - l_in                                     # GRV front face
        x_can = L - length_with_radius(pr, L, self.R, CANSAT_R)  # first station wide enough for the CanSat
        return dict(l_in=l_in, protrusion=lg - l_in - c, x_grv=x_grv, x_can=x_can,
                    cansat_room=x_grv - p.axial_clearance - x_can, cansat_room_all=L + c - x_can)

    def coupler_for(self, shape: str, limit: float, total: float = TOTAL, l_grv: float | None = None) -> float:
        """Shortest coupler >= C_MIN whose GRV protrusion is <= limit (rounded up to 1 mm)."""
        if not math.isfinite(limit):
            return C_MIN
        g = lambda c: self.fit(shape, total - c, c, l_grv)["protrusion"] - limit  # noqa: E731
        if g(C_MIN) <= 0:
            return C_MIN
        c = brentq(g, C_MIN, total - 0.05)                   # protrusion falls as c grows
        return math.ceil(round(c * 1e3, 6)) / 1e3

    # ---------------------------------------------------------------- flight
    def vehicle(self, shape: str, L: float, c: float):
        """Drag model: curved nose L, then the coupler cylinder, then the fixed rest of the vehicle."""
        veh = self.st.vehicle(shape, L / (2 * self.R))
        veh.L_body += c
        veh.L += c
        veh.S_body += np.pi * self.cfg.vehicle.diameter * c
        return veh

    def nose_mass(self, shape: str, L: float, c: float) -> float:
        """Shell of the curved part and of the coupler cylinder, plus the tip insert [kg]."""
        s = self.cfg.structure
        return self.st.nose_mass(shape, L / (2 * self.R)) + s.shell_areal_density * np.pi * self.cfg.vehicle.diameter * c

    def mass_delta(self, shape: str, L: float, c: float) -> float:
        return self.nose_mass(shape, L, c) - self.st.nose_mass(*self.st.baseline)

    def trajectory(self, shape: str, L: float, c: float, envelope: bool = False, detail: bool = False) -> dict:
        fl = self.cfg.flight
        kw = dict(m_nose_delta=self.mass_delta(shape, L, c))
        if detail:
            kw["dt_out"] = fl.detail_output_step
        if envelope:
            return simulate(self.cfg, self.vehicle(shape, L, c), mass_model=fl.envelope_mass_model,
                            ground_temperature=fl.envelope_ground_temperature, **kw)
        return simulate(self.cfg, self.vehicle(shape, L, c), mass_model=fl.mass_model, **kw)

    def apogee(self, shape: str, L: float, c: float) -> float:
        return float(self.trajectory(shape, L, c)["apogee"])

    def burnout_mass(self) -> float:
        e, v = self.cfg.engine, self.cfg.vehicle
        return v.mass_wet - e.total_impulse / (e.specific_impulse * NC.G0)

    # ---------------------------------------------------------------- stability
    def d_static_margin(self, shape: str, L: float, c: float) -> float:
        """Static-margin change [cal] vs the baseline nose (positions from the aft end, as report 1)."""
        cfg, R = self.cfg, self.R
        D = 2 * R
        cna_f, xf = fin_cna_xcp(cfg)
        L_body = cfg.vehicle.length - cfg.vehicle.reference_nose_length
        x_f_aft = cfg.fins.root_chord - xf
        rho_a, frac = cfg.structure.shell_areal_density, cfg.structure.shell_cg_fraction

        def cp_cg(pr, L, c):
            V, _ = volume_and_area(pr, L, R)
            base = L_body + c                                   # nose base (shoulder) from the aft end
            x_cp = (2 * (base + V / (np.pi * R**2)) + cna_f * x_f_aft) / (2 + cna_f)
            m_n = self.st.nose_mass(pr.name, L / D)
            m_c = rho_a * np.pi * D * c
            x_m = (m_n * (base + frac * L) + m_c * (L_body + c / 2)) / (m_n + m_c)
            return x_cp, m_n + m_c, x_m

        b_shape, b_f = self.st.baseline
        cp0, m0, x0 = cp_cg(self.pr(b_shape), 2 * R * b_f, 0.0)
        cp1, m1, x1 = cp_cg(self.pr(shape), L, c)
        dx_cg = (m1 * x1 - m0 * x0) / (self.burnout_mass() + m1 - m0)
        return float((dx_cg - (cp1 - cp0)) / D)

    # ---------------------------------------------------------------- loads at the joint
    def joint_loads(self, shape: str, L: float, c: float, tr: dict, envelope: bool, m_pay: float) -> dict:
        """Axial force on the nose joint (compression > 0) and aerodynamic bending per degree.

        Nose free body along the axis: F = m_n (T - D)/m + D_n, with m_n the nose with its payload
        structure (Req_04 limit) and the payload it carries, D the vehicle drag and D_n the nose drag
        (wave drag plus the nose share of the skin friction, the latter unscaled by k: conservative).
        """
        cfg, R = self.cfg, self.R
        e, v, fl = cfg.engine, cfg.vehicle, cfg.flight
        veh = self.vehicle(shape, L, c)
        t, M, q, rho, mu, V = tr["t"], tr["M"], tr["q"], tr["rho"], tr["mu"], tr["V"]
        Re = np.where(V > 1.0, rho * V / mu, rho * 1.0 / mu)  # q ~ 0 there anyway
        burning = t < e.burn_time
        cd = np.where(burning, veh.cd(M, Re, True), veh.cd(M, Re, False))
        D = q * veh.Aref * cd
        mm = fl.envelope_mass_model if envelope else fl.mass_model
        m0 = v.mass_wet + self.mass_delta(shape, L, c)
        mdot = (v.mass_wet - v.mass_dry) / e.burn_time if mm == "lvo" else \
            e.total_impulse / (e.specific_impulse * NC.G0) / e.burn_time
        m = m0 - mdot * np.minimum(t, e.burn_time)
        T = np.where(burning, e.thrust, 0.0)
        _, S_n = volume_and_area(self.pr(shape), L, R)
        S_n += np.pi * v.diameter * c
        cf = skin_friction(Re * veh.L, M, veh.L, veh.roughness)
        cd_n = veh.cdw(M) + cf * (1 + 1 / (2 * veh.L / veh.D)) * S_n / veh.Aref
        D_n = q * veh.Aref * cd_n
        m_n = STRUCT_MAX + m_pay
        F = m_n * (T - D) / m + D_n
        V_n, _ = volume_and_area(self.pr(shape), L, R)
        arm = (L + c) - (L - V_n / (np.pi * R**2))            # nose CP to the joint plane
        i, j = int(np.argmax(q)), int(np.argmax(F))
        return dict(F_max=float(F[j]), t_F_max=float(t[j]), F_drag=float(D_n[j]), F_min=float(F.min()),
                    F_liftoff=float(F[0]), q_max=float(q[i]), M_q_max=float(M[i]),
                    D_nose_max=float(D_n.max()), arm=float(arm), m_nose=m_n,
                    req07=float(G_REQ07 * NC.G0 * m_n * FOS),
                    bend_per_deg=float(2.0 * np.radians(1.0) * q[i] * veh.Aref * arm))

    # ---------------------------------------------------------------- heating
    def heating(self, shape: str, L: float, nom: dict, env: dict) -> dict:
        """Peak temperatures [degC]: tip insert, shell inner face at 50 mm, mid-length and the base."""
        h, cfg = self.cfg.heating, self.cfg
        mats = self.st.materials()
        pr = self.pr(shape)
        out = dict(T0=[float(np.max(ht.stagnation_state(tr["M"], tr["p"], tr["T"])[1])) - 273.15 for tr in (nom, env)],
                   tip=[float(ht.tip_insert(tr, pr, L, self.R, self.rn, h.tip_length, cfg.structure.tip_mass,
                                            mats[h.tip_material])["T"].max()) - 273.15 for tr in (nom, env)])
        for key, x in (("shell_50", 0.05), ("shell_mid", 0.5 * L), ("shell_base", L)):
            out[key] = [float(ht.wall_response(tr, mats[h.shell_material], h.shell_thickness, x,
                                               n=h.wall_nodes)["T_in"].max()) - 273.15 for tr in (nom, env)]
        return out


# ------------------------------------------------------------------------------- evaluation
def pick(g: pd.DataFrame) -> pd.Series:
    """Row of highest apogee; within TIE of it, the first profile of PREF."""
    ok = g[g.apogee >= g.apogee.max() - TIE]
    return ok.loc[ok["shape"].map(PREF.index).idxmin()]


def sweep(mdl: Model) -> pd.DataFrame:
    rows = []
    for shape in PROFILES:
        for c in C_SWEEP:
            L = TOTAL - c
            if L / (2 * mdl.R) < MIN_FINENESS - 1e-9:
                continue
            f = mdl.fit(shape, L, c)
            rows.append(dict(shape=shape, c_mm=1e3 * c, L_mm=1e3 * L, fineness=L / (2 * mdl.R),
                             apogee=mdl.apogee(shape, L, c), l_in_mm=1e3 * f["l_in"],
                             protrusion_mm=1e3 * f["protrusion"]))
    return pd.DataFrame(rows)


def total_sweep(mdl: Model) -> pd.DataFrame:
    """Design of each profile in each study for each allowed total length (interface meeting)."""
    rows = []
    for rule, lim in RULES.items():
        for shape in PROFILES:
            for T in T_SWEEP:
                c = mdl.coupler_for(shape, lim, total=T)
                L = T - c
                if L / (2 * mdl.R) < MIN_FINENESS - 1e-9:
                    continue
                rows.append(dict(rule=rule, shape=shape, total_mm=1e3 * T, c_mm=1e3 * c, L_mm=1e3 * L,
                                 apogee=mdl.apogee(shape, L, c)))
    return pd.DataFrame(rows)


def design_points(mdl: Model) -> pd.DataFrame:
    rows = []
    for rule, lim in RULES.items():
        for shape in PROFILES:
            c = mdl.coupler_for(shape, lim)
            L = TOTAL - c
            f = mdl.fit(shape, L, c)
            rows.append(dict(rule=rule, shape=shape, c_mm=1e3 * c, L_mm=1e3 * L, fineness=L / (2 * mdl.R),
                             apogee=mdl.apogee(shape, L, c), l_in_mm=1e3 * f["l_in"],
                             in_coupler_mm=1e3 * min(c, max(mdl.l_grv - f["l_in"], 0.0)),
                             protrusion_mm=1e3 * f["protrusion"],
                             cansat_room_mm=1e3 * (f["cansat_room"] if rule == "inside" else f["cansat_room_all"]),
                             mass_g=1e3 * mdl.nose_mass(shape, L, c),
                             dSM=mdl.d_static_margin(shape, L, c)))
    d = pd.DataFrame(rows)
    return flag(d, mdl.cfg.flight.target_apogee)


def flag(d: pd.DataFrame, target: float) -> pd.DataFrame:
    d["d9km"] = d.apogee - target
    d["best"] = d.groupby("rule").apogee.transform("max") - d.apogee <= TIE
    d["design"] = False
    for rule in RULES:
        d.loc[pick(d[d.rule == rule]).name, "design"] = True
    return d


def nine_km(ts: pd.DataFrame, rule: str, target: float) -> float:
    """Allowed total length [mm] at which the best profile of a study reaches the target."""
    env = ts[ts.rule == rule].groupby("total_mm").apogee.max()
    above = env[env >= target]
    if not len(above):
        return math.nan
    i = list(env.index).index(above.index[0])
    if i == 0:
        return float(env.index[0])
    t0, t1, z0, z1 = env.index[i - 1], env.index[i], env.iloc[i - 1], env.iloc[i]
    return float(t0 + (target - z0) * (t1 - t0) / (z1 - z0))


def cfd_check(mdl: Model) -> pd.DataFrame:
    """New direct Euler values at L/D 2.5 and 3.5 against the report-1 database model."""
    from nosecone import aerodynamics as aero
    db = mdl.st.db
    old = aero.select(db, ["main", "f45"], 0.0, mdl.cfg.aerodynamics.max_fineness, rn=0.0)
    new = db[(db.study == "f650") & np.isclose(db.rn, 0.0)]
    rows = []
    for shape in PROFILES:
        for f in sorted(new[new["shape"] == shape].fineness.unique()):
            w = aero.WaveDrag(old, shape, f, mdl.cfg.aerodynamics.machs, rn=0.0, grid=mdl.cfg.aerodynamics.grid)
            for M in sorted(new[(new["shape"] == shape) & (new.fineness == f)].mach):
                cd = float(new[(new["shape"] == shape) & (new.fineness == f) & (new.mach == M)].cd.iloc[0])
                rows.append(dict(shape=shape, fineness=f, mach=M, cd_direct=cd, cd_report1=float(w.cd_sup[
                    list(w.machs).index(M)]) if M in w.machs else np.nan))
    d = pd.DataFrame(rows, columns=["shape", "fineness", "mach", "cd_direct", "cd_report1"])
    d["err_pct"] = 100 * (d.cd_report1 / d.cd_direct - 1)
    return d


def export_cad(mdl: Model, d: pd.DataFrame, outdir: Path) -> dict:
    cad = mdl.cfg.cad
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for _, r in d.iterrows():
        pr, L, c = mdl.pr(r["shape"]), 1e-3 * r.L_mm, 1e-3 * r.c_mm
        stem = f"nose_{r['shape']}_L{r.L_mm:.0f}_c{r.c_mm:.0f}_rn{1e3 * mdl.rn:g}mm"
        key = f"{r.rule}:{r['shape']}"
        if any(v["stem"] == stem for v in res.values()):         # same geometry in both studies
            res[key] = next(v for v in res.values() if v["stem"] == stem)
            continue
        L_phys = export.write_profile_csv(outdir / f"{stem}.csv", pr, L, mdl.R, mdl.rn, cad.profile_points,
                                          cad.profile_clustering)
        x, rr = export.outline(pr, L, mdl.R, c, rn=mdl.rn, n=cad.stl_points_design)
        export.write_stl(outdir / f"{stem}.stl", x, rr, n_az=cad.stl_facets, name=f"{r['shape']} L={r.L_mm:.0f} c={r.c_mm:.0f}")
        stl = export.stl_check(outdir / f"{stem}.stl")
        xs, rs = export.profile_points(pr, L, mdl.R, mdl.rn, cad.step_points, cad.profile_clustering)
        step = export.write_step(outdir / f"{stem}.step", 1e3 * xs, 1e3 * rs, 1e3 * c, name=r["shape"])
        res[key] = dict(shape=r["shape"], stem=stem, L_phys_mm=L_phys, total_phys_mm=L_phys + r.c_mm,
                        stl_open_edges=stl["open_edges"], stl_volume_L=stl["volume_mm3"] / 1e6,
                        step_volume_L=None if step is None else step["volume_mm3"] / 1e6)
    return res


# ---------------------------------------------------------------------------------- figures
def figures(mdl: Model, sw: pd.DataFrame, ts: pd.DataFrame, d: pd.DataFrame, figdir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from nosecone.plotting import style as fs

    fs.setup()
    figdir.mkdir(parents=True, exist_ok=True)
    tgt = mdl.cfg.flight.target_apogee
    rmark = {"inside": "o", "free": "^"}
    t_lv = 1e3 * (LV_MAX - mdl.rest)

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(figdir / f"{name}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)

    # (a) apogee and (b) GRV protrusion against the coupler length
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.42 * fs.WIDTH))
    for shape in PROFILES:
        s = sw[sw["shape"] == shape]
        kw = dict(color=fs.COLORS[shape], ls=fs.STYLES[shape])
        a.plot(s.c_mm, s.apogee, **kw)
        b.plot(s.c_mm, s.protrusion_mm, **kw)
        for ru, mk in rmark.items():
            r = d[(d.rule == ru) & (d["shape"] == shape)].iloc[0]
            mfc = fs.COLORS[shape] if r.design else "white"
            for ax, y in ((a, r.apogee), (b, r.protrusion_mm)):
                ax.plot(r.c_mm, y, marker=mk, color=fs.COLORS[shape], mfc=mfc, ms=4.5, ls="none", zorder=5)
    a.axhline(tgt, color=fs.INK2, lw=0.6, ls=(0, (4, 2)))
    a.text(196, tgt - 8, r"\SI{9}{km} (SE\_REQ\_01)", ha="right", va="top", fontsize=7, color=fs.INK2)
    a.set(xlabel=r"Coupler length $c$ [mm]", ylabel=r"Apogee [m]", xlim=(15, 205))
    b.axhspan(-200, 0, color="#e6f2e6", lw=0)
    b.axhline(0, color=fs.INK2, lw=0.6)
    b.legend(handles=[Patch(color="#e6f2e6", label="3U inside the nose")], loc="upper right", fontsize=7,
             handlelength=1.4)
    b.set(xlabel=r"Coupler length $c$ [mm]", ylabel=r"3U beyond the nose joint $p$ [mm]", xlim=(15, 205),
          ylim=(-60, 160))
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        top = ax.secondary_xaxis("top", functions=(lambda c: 1e3 * TOTAL - c, lambda L: 1e3 * TOTAL - L))
        top.set_xlabel(r"Curved length $L = 650 - c$ [mm]")
        ax.text(0.03, 0.04, lab, transform=ax.transAxes, fontsize=9)
    h = [Line2D([], [], color=fs.COLORS[s], ls=fs.STYLES[s], label=mdl.pr(s).label) for s in PROFILES]
    h += [Line2D([], [], color=fs.INK, marker=rmark[ru], mfc="white", ls="none", label=f"Study {RULE_LABELS[ru]}")
          for ru in RULES]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7, handlelength=2.4,
               columnspacing=1.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_tl_coupler")

    # apogee against the allowed total length, one panel per study
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.40 * fs.WIDTH), sharey=True)
    for ax, ru, lab in zip(axs, RULES, ("(a)", "(b)")):
        t = ts[ts.rule == ru]
        for shape in PROFILES:
            s = t[t["shape"] == shape]
            ax.plot(s.total_mm, s.apogee, color=fs.COLORS[shape], ls=fs.STYLES[shape], label=mdl.pr(shape).label)
        r = d[(d.rule == ru) & d.design].iloc[0]
        ax.plot(1e3 * TOTAL, r.apogee, marker=rmark[ru], color=fs.COLORS[r["shape"]], ms=5, ls="none", zorder=5)
        ax.axhline(tgt, color=fs.INK2, lw=0.6, ls=(0, (4, 2)))
        ax.axvline(1e3 * TOTAL, color=fs.INK2, lw=0.6)
        ax.axvline(t_lv, color=fs.INK2, lw=0.6, ls=(0, (1, 1.5)))
        ax.set_title(f"Study {RULE_LABELS[ru]}", fontsize=8)
        ax.set_xlabel(r"Total nose length $L + c$ [mm]")
        ax.set_xlim(600, 850)
        ax.text(0.03, 0.96, lab, transform=ax.transAxes, fontsize=9, va="top")
    axs[0].set_ylabel("Apogee [m]")
    axs[0].text(670, tgt + 8, r"\SI{9}{km}", ha="left", va="bottom", fontsize=7, color=fs.INK2)
    h = axs[1].get_legend_handles_labels()[0]
    h += [Line2D([], [], color=fs.INK2, lw=0.6, label=r"Req\_03 (650 mm)"),
          Line2D([], [], color=fs.INK2, lw=0.6, ls=(0, (1, 1.5)), label=rf"LV \SI{{5.4}}{{m}} ({t_lv:.0f} mm)")]
    axs[1].legend(handles=h, loc="lower right", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_tl_total")

    # the two designs to scale, stacked in one equal-aspect axes (mm, x from the apex)
    sel = [d[(d.rule == ru) & d.design].iloc[0] for ru in RULES]
    Rm, gap = 1e3 * mdl.R, 250.0
    p = mdl.cfg.payload
    span_y = gap * (len(sel) - 1) + 2 * Rm + 80
    span_x = 1e3 * TOTAL + 150
    fig, ax = plt.subplots(figsize=(fs.WIDTH, fs.WIDTH * span_y / span_x))
    for i, r in enumerate(sel):
        y0, shape = -gap * i, r["shape"]
        pr, L, c = mdl.pr(shape), 1e-3 * r.L_mm, 1e-3 * r.c_mm
        x, y = surface(pr, L, mdl.R, rn=mdl.rn, n=1500)
        X, Y = 1e3 * x, 1e3 * y
        for sg in (1, -1):
            ax.plot(X, y0 + sg * Y, color=fs.INK, lw=0.9)
            ax.plot([X[-1], X[-1] + r.c_mm], [y0 + sg * Rm] * 2, color=fs.INK, lw=0.9)
        ax.fill_between(X, y0 - Y, y0 + Y, color=fs.COLORS[shape], alpha=0.12, lw=0)
        ax.add_patch(plt.Rectangle((X[-1], y0 - Rm), r.c_mm, 2 * Rm, color=fs.INK2, alpha=0.20, lw=0))
        ax.plot([X[-1]] * 2, [y0 - Rm, y0 + Rm], color=fs.INK2, lw=0.5, ls=(0, (3, 2)))
        xj = X[-1] + r.c_mm
        ax.plot([xj] * 2, [y0 - Rm - 10, y0 + Rm + 10], color=fs.INK, lw=1.1)
        ax.text(xj + 6, y0 - Rm, "joint\nto RE", ha="left", va="bottom", fontsize=6.5)
        xt = 1e3 * mdl.cfg.heating.tip_length
        s = X <= xt
        ax.fill_between(X[s], y0 - Y[s], y0 + Y[s], color="#a7b0b8", lw=0)
        f = mdl.fit(shape, L, c)
        xg = 1e3 * f["x_grv"]
        if r.rule == "inside":
            ax.add_patch(plt.Rectangle((xg, y0 - 50), 1e3 * p.length, 100, fill=False, ec="#eb6834", lw=0.9))
            ax.text(xg + 0.5e3 * p.length, y0, r"3U (GRV) $100\times100\times300$", ha="center", va="center",
                    fontsize=6.5)
            xc = xg - 1e3 * p.axial_clearance - 1e3 * CANSAT_H
        else:
            ax.add_patch(plt.Rectangle((xg, y0 - 50), 1e3 * p.length, 100, fill=False, ec=fs.INK2, lw=0.7,
                                       ls=(0, (1, 1.5))))
            ax.text(0.5 * (xg + xj), y0, f"3U if it were here:\n{r.protrusion_mm:.0f} mm past the joint",
                    ha="center", va="center", fontsize=6.5, color=fs.INK2)
            xc = 1e3 * f["x_can"]
        ax.add_patch(plt.Rectangle((xc, y0 - 1e3 * CANSAT_D / 2), 1e3 * CANSAT_H, 1e3 * CANSAT_D, fill=False,
                                   ec="#2a78d6", lw=0.9, ls=(0, (3, 1.5))))
        ax.text(xc + 0.5e3 * CANSAT_H, y0, "CanSat", ha="center", va="center", fontsize=6.5)
        ax.text(0, y0 + Rm + 8, rf"\textbf{{Study {RULE_LABELS[r.rule]}}} -- {mdl.pr(shape).label}: "
                rf"curved part {r.L_mm:.0f} mm + coupler {r.c_mm:.0f} mm $=$ 650 mm, apogee {r.apogee:.0f} m",
                ha="left", va="bottom", fontsize=7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set(xlim=(-10, span_x - 10), ylim=(-gap * (len(sel) - 1) - Rm - 25, Rm + 35))
    save(fig, "fig_tl_designs")


# ----------------------------------------------------------------------------------- LaTeX
def num(v: float, nd: int = 0) -> str:
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return s.replace("-", "$-$") if v < 0 else s


def sgn(v: float, nd: int = 0) -> str:
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.{nd}f}"


def write_latex(mdl: Model, d: pd.DataFrame, ts: pd.DataFrame, cfdc: pd.DataFrame, loads: dict, heat: dict, cad: dict,
                sens: pd.DataFrame, dz_dm: float, secdir: Path) -> dict:
    secdir.mkdir(parents=True, exist_ok=True)
    tgt = mdl.cfg.flight.target_apogee
    des = {ru: d[(d.rule == ru) & d.design].iloc[0] for ru in RULES}
    # design points
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Study & Profile & $c$ & $L$ & $L/D$ & 3U in $L$ & $p$ & Apogee & vs \SI{9}{km} \\",
             r" & & [mm] & [mm] & & [mm] & [mm] & [m] & [m] \\", r"\midrule"]
    for ru in RULES:
        dd = d[d.rule == ru].set_index("shape").loc[PROFILES]
        for i, (shape, r) in enumerate(dd.iterrows()):
            b = (lambda s: rf"\textbf{{{s}}}") if r.best else (lambda s: s)
            rl = RULE_LABELS[ru] if i == 0 else ""
            lines.append(f"{rl} & {b(LABELS[shape])} & {r.c_mm:.0f} & {r.L_mm:.0f} & {r.fineness:.2f} & "
                         f"{r.l_in_mm:.0f} & {num(r.protrusion_mm)} & {b(f'{r.apogee:.0f}')} & {b(sgn(r.d9km))} \\\\")
        lines.append(r"\midrule" if ru != list(RULES)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (secdir / "tab_tl_designs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # what a longer nose buys (interface meeting)
    lines = [r"\begin{tabular}{@{}rrlrrlrr@{}}", r"\toprule",
             r"Total & LV & \multicolumn{3}{c}{Study A: 3U in the nose} & \multicolumn{3}{c}{Study B: 3U not in the nose} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"$L+c$ [mm] & [m] & Design $L+c$ & Apogee [m] & Gain [m] & Design $L+c$ & Apogee [m] & Gain [m] \\",
             r"\midrule"]
    base = {ru: des[ru].apogee for ru in RULES}
    for T in T_TABLE:
        cells = []
        for ru in RULES:
            g = ts[(ts.rule == ru) & np.isclose(ts.total_mm, T)]
            r = pick(g) if abs(T - 1e3 * TOTAL) > 1e-6 else des[ru]
            cells.append(f"{PKEY[r['shape']]} {r.L_mm:.0f}\\,+\\,{r.c_mm:.0f} & {r.apogee:.0f} & "
                         f"{sgn(r.apogee - base[ru]) if T != 1e3 * TOTAL else '--'}")
        lv = mdl.rest + 1e-3 * T
        lvs = f"{lv:.2f}" if lv < LV_MAX - 1e-9 else rf"\textit{{{lv:.2f}}}"
        lines.append(f"{T:.0f} & {lvs} & " + " & ".join(cells) + r" \\")
    n9 = {ru: nine_km(ts, ru, tgt) for ru in RULES}
    lines += [r"\midrule",
              r"\multicolumn{2}{@{}l}{\SI{9}{km} reached at} & \multicolumn{3}{l}{" +
              rf"$L+c\approx\SI{{{5 * round(n9['inside'] / 5):.0f}}}{{mm}}$ (LV \SI{{{mdl.rest + 1e-3 * n9['inside']:.2f}}}{{m}})}} & " +
              r"\multicolumn{3}{l}{" +
              rf"$L+c\approx\SI{{{5 * round(n9['free'] / 5):.0f}}}{{mm}}$ (LV \SI{{{mdl.rest + 1e-3 * n9['free']:.2f}}}{{m}})}} \\",
              r"\bottomrule", r"\end{tabular}"]
    (secdir / "tab_tl_length.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CFD check
    piv = cfdc.pivot_table(index=["shape", "fineness"], columns="mach", values="cd_direct")
    err = cfdc.pivot_table(index=["shape", "fineness"], columns="mach", values="err_pct")
    machs = list(piv.columns)
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(machs) + "r@{}}", r"\toprule",
             r"Profile & $L/D$ & " + " & ".join(f"$M={m:g}$" for m in machs) + r" & report-1 model at $M=1.4$ \\",
             r"\midrule"]
    for shape in PROFILES:
        for f in piv.loc[shape].index:
            vals = " & ".join(f"{piv.loc[(shape, f), m]:.4f}" for m in machs)
            e = err.loc[(shape, f), 1.4]
            lines.append(f"{LABELS[shape]} & {f:g} & {vals} & {sgn(e, 1)}\\,\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (secdir / "tab_tl_cfd.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # loads and heating of the two designs
    keys = [f"{ru}:{des[ru]['shape']}" for ru in RULES]
    hdr = " & ".join(rf"{STUDY[ru]}: {LABELS[des[ru]['shape']]}" for ru in RULES)
    L_ = lambda key, fmt: " & ".join(fmt(loads[k][key]) for k in keys)  # noqa: E731
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", rf"Quantity & {hdr} \\", r"\midrule",
             r"Nose mass in the load case (Req\_04 limit + payload) [kg] & " + L_("m_nose", lambda v: f"{v:.2f}") + r" \\",
             rf"Maximum dynamic pressure $q_{{\max}}$ [kPa] & " + L_("q_max", lambda v: f"{v / 1e3:.0f}") + r" \\",
             r"Nose drag at its maximum [N] & " + L_("D_nose_max", lambda v: f"{v:.0f}") + r" \\",
             r"Joint compression at lift-off [N] & " + L_("F_liftoff", lambda v: f"{v:.0f}") + r" \\",
             r"Joint compression, maximum (limit) [N] & " + L_("F_max", lambda v: f"{v:.0f}") + r" \\",
             r"\quad of which nose drag [N] & " + L_("F_drag", lambda v: f"{v:.0f}") + r" \\",
             r"\quad time of the maximum [s] & " + L_("t_F_max", lambda v: f"{v:.1f}") + r" \\",
             rf"\quad ultimate ($\times{FOS:g}$) vs Req\_06 (\SI{{{REQ06_N:.0f}}}{{N}}) [N] & " + L_("F_max", lambda v: f"{FOS * v:.0f}") + r" \\",
             rf"Traction, Req\_07 rationale ({G_REQ07:g}\,$g$, $\times{FOS:g}$) vs \SI{{2000}}{{N}} [N] & " + L_("req07", lambda v: f"{v:.0f}") + r" \\",
             r"Nose CP to the joint plane [mm] & " + L_("arm", lambda v: f"{1e3 * v:.0f}") + r" \\",
             rf"Bending at the joint per degree at $q_{{\max}}$ (limit) [N\,m/$^\circ$] & " + L_("bend_per_deg", lambda v: f"{v:.1f}") + r" \\",
             rf"Incidence at which Req\_12 (\SI{{{REQ12_NM:.0f}}}{{N.m}} ultimate) is reached [$^\circ$] & " + L_("bend_per_deg", lambda v: f"{REQ12_NM / FOS / v:.1f}") + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    (secdir / "tab_tl_loads.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    H = lambda key: " & ".join(f"{heat[k][key][0]:.0f} / {heat[k][key][1]:.0f}" for k in keys)  # noqa: E731
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", rf"Peak temperature [\si{{\celsius}}], nominal / envelope & {hdr} \\",
             r"\midrule",
             r"Stagnation temperature $T_0$ & " + H("T0") + r" \\",
             r"Al 6082 tip insert & " + H("tip") + r" \\",
             r"Shell inner face, \SI{50}{mm} from the tip & " + H("shell_50") + r" \\",
             r"Shell inner face, mid-length & " + H("shell_mid") + r" \\",
             r"Shell inner face at the coupler (bond) & " + H("shell_base") + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    (secdir / "tab_tl_thermal.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # macros: TL{In|Out}{VK|LVH|PH}{field} for every design point
    m = {}
    for _, r in d.iterrows():
        s, k = r["shape"], f"{r.rule}:{r['shape']}"
        key = f"TL{SKEY[r.rule]}{PKEY[s]}"
        ld, hh, cd = loads[k], heat[k], cad[k]
        trunc = r.L_mm - cd["L_phys_mm"]                   # apex removed by the spherical tip
        m.update({
            f"{key}C": rf"\SI{{{r.c_mm:.0f}}}{{mm}}",
            f"{key}L": rf"\SI{{{r.L_mm:.0f}}}{{mm}}",
            f"{key}LD": f"{r.fineness:.2f}",
            f"{key}APO": rf"\SI{{{r.apogee:.0f}}}{{m}}",
            f"{key}D": rf"\SI{{{r.d9km:+.0f}}}{{m}}",
            f"{key}P": rf"\SI{{{r.protrusion_mm:.0f}}}{{mm}}",
            f"{key}LIN": rf"\SI{{{r.l_in_mm:.0f}}}{{mm}}",
            f"{key}INC": rf"\SI{{{r.in_coupler_mm:.0f}}}{{mm}}",
            f"{key}CAN": rf"\SI{{{r.cansat_room_mm:.0f}}}{{mm}}",
            f"{key}MASS": rf"\SI{{{r.mass_g:.0f}}}{{g}}",
            f"{key}FDMASS": rf"\SI{{{r.mass_g + 1e3 * EXTRA_STRUCT:.0f}}}{{g}}",
            f"{key}LEFT": rf"\SI{{{1e3 * STRUCT_MAX - r.mass_g - 1e3 * EXTRA_STRUCT:.0f}}}{{g}}",
            f"{key}DSM": rf"{r.dSM:+.3f}".replace("-", "$-$").replace("+", "$+$"),
            f"{key}MNOSE": rf"\SI{{{ld['m_nose']:.2f}}}{{kg}}",
            f"{key}FMAX": rf"\SI{{{ld['F_max']:.0f}}}{{N}}",
            f"{key}FULT": rf"\SI{{{FOS * ld['F_max']:.0f}}}{{N}}",
            f"{key}FLIFT": rf"\SI{{{ld['F_liftoff']:.0f}}}{{N}}",
            f"{key}FDRAG": rf"\SI{{{ld['F_drag']:.0f}}}{{N}}",
            f"{key}FDRAGPCT": rf"\SI{{{100 * ld['F_drag'] / ld['F_max']:.0f}}}{{\percent}}",
            f"{key}REQSEVEN": rf"\SI{{{ld['req07'] / 1e3:.2f}}}{{kN}}",
            f"{key}BEND": rf"\SI{{{ld['bend_per_deg']:.0f}}}{{N.m}}",
            f"{key}ALPHA": rf"\ang{{{REQ12_NM / FOS / ld['bend_per_deg']:.1f}}}",
            f"{key}QMAX": rf"\SI{{{ld['q_max'] / 1e3:.0f}}}{{kPa}}",
            f"{key}MQ": f"{ld['M_q_max']:.2f}",
            f"{key}ARM": rf"\SI{{{1e3 * ld['arm']:.0f}}}{{mm}}",
            f"{key}XCP": rf"\SI{{{r.L_mm + r.c_mm - 1e3 * ld['arm'] - trunc:.0f}}}{{mm}}",
            f"{key}MMAX": f"{ld['M_max'][0]:.2f}",
            f"{key}MMAXENV": f"{ld['M_max'][1]:.2f}",
            f"{key}TIP": rf"\SI{{{hh['tip'][1]:.0f}}}{{\celsius}}",
            f"{key}BOND": rf"\SI{{{hh['shell_base'][1]:.0f}}}{{\celsius}}",
            f"{key}BONDNOM": rf"\SI{{{hh['shell_base'][0]:.0f}}}{{\celsius}}",
            f"{key}SHELL": rf"\SI{{{max(hh['shell_50'][1], hh['shell_mid'][1]):.0f}}}{{\celsius}}",
            f"{key}SHELLNOM": rf"\SI{{{max(hh['shell_50'][0], hh['shell_mid'][0]):.0f}}}{{\celsius}}",
            f"{key}TZERO": rf"\SI{{{hh['T0'][1] + 273.15:.0f}}}{{K}}",
            f"{key}TZEROC": rf"\SI{{{hh['T0'][1]:.0f}}}{{\celsius}}",
            f"{key}FILE": r"\texttt{" + cd["stem"].replace("_", r"\_") + r"}",
            f"{key}PHYS": rf"\SI{{{cd['L_phys_mm']:.1f}}}{{mm}}",
            f"{key}TOTPHYS": rf"\SI{{{cd['total_phys_mm']:.1f}}}{{mm}}",
            f"{key}TRUNC": rf"\SI{{{trunc:.1f}}}{{mm}}",
            f"{key}XGRV": rf"\SI{{{r.L_mm - r.l_in_mm - trunc:.0f}}}{{mm}}",
        })
    A, B = des["inside"], des["free"]
    ins = d[d.rule == "inside"].set_index("shape")
    fr = d[d.rule == "free"].set_index("shape")
    m["TLGAPVK"] = rf"\SI{{{A.apogee - ins.loc['vonkarman'].apogee:.0f}}}{{m}}"
    m["TLGAPPH"] = rf"\SI{{{A.apogee - ins.loc['power050'].apogee:.0f}}}{{m}}"
    m["TLEQGAP"] = rf"\SI{{{fr.loc['vonkarman'].apogee - fr.loc['lvhaack'].apogee:.0f}}}{{m}}"
    m["TLEQGAPPH"] = rf"\SI{{{fr.loc['power050'].apogee - fr.loc['vonkarman'].apogee:.0f}}}{{m}}"
    m["TLAB"] = rf"\SI{{{B.apogee - A.apogee:.0f}}}{{m}}"
    m["TLDZDM"] = rf"\SI{{{abs(dz_dm):.0f}}}{{m/kg}}"
    m["TLABKG"] = rf"\SI{{{(B.apogee - A.apogee) / abs(dz_dm):.1f}}}{{kg}}"
    m["TLInKG"] = rf"\SI{{{-A.d9km / abs(dz_dm):.1f}}}{{kg}}"
    m["TLOutKG"] = rf"\SI{{{-B.d9km / abs(dz_dm):.1f}}}{{kg}}"
    m["TLGRVLEN"] = rf"\SI{{{1e3 * mdl.cfg.payload.length:.0f}}}{{mm}}"
    m["TLCANMASS"] = rf"\SI{{{CANSAT_MASS:.2f}}}{{kg}}"
    m["TLCMIN"] = rf"\SI{{{1e3 * C_MIN:.0f}}}{{mm}}"
    m["TLREST"] = rf"\SI{{{mdl.rest:.2f}}}{{m}}"
    m["TLLVSIX"] = rf"\SI{{{mdl.rest + TOTAL:.2f}}}{{m}}"
    m["TLLVNOSE"] = rf"\SI{{{1e3 * (LV_MAX - mdl.rest):.0f}}}{{mm}}"
    m["TLGRVSM"] = rf"{GRV_MASS * 0.1 / (mdl.burnout_mass() * 2 * mdl.R):.2f}"
    for ru in RULES:
        g = ts[ts.rule == ru]
        z = {T: pick(g[np.isclose(g.total_mm, T)]).apogee for T in (700.0, 720.0, 750.0)}
        z[650.0] = des[ru].apogee
        p = SKEY[ru]
        m[f"TL{p}SLOPE"] = rf"\SI{{{(z[700.0] - z[650.0]) / 50:.1f}}}{{m/mm}}"
        m[f"TL{p}SEVEN"] = rf"\SI{{{z[700.0]:.0f}}}{{m}}"
        m[f"TL{p}SEVENTWENTY"] = rf"\SI{{{z[720.0]:.0f}}}{{m}}"
        m[f"TL{p}SEVENTWENTYG"] = rf"\SI{{{z[720.0] - z[650.0]:+.0f}}}{{m}}"
        m[f"TL{p}SHARE"] = rf"\SI{{{100 * (z[720.0] - z[650.0]) / -des[ru].d9km:.0f}}}{{\percent}}"
        m[f"TL{p}SEVENFIFTY"] = rf"\SI{{{z[750.0]:.0f}}}{{m}}"
        n9 = nine_km(ts, ru, tgt)
        m[f"TL{p}NINE"] = rf"\SI{{{5 * round(n9 / 5):.0f}}}{{mm}}" if math.isfinite(n9) else r"more than \SI{850}{mm}"
        m[f"TL{p}NINELV"] = rf"\SI{{{mdl.rest + 1e-3 * n9:.2f}}}{{m}}" if math.isfinite(n9) else "--"
        m[f"TL{p}NINESHAPE"] = LABELS[pick(g[g.total_mm >= n9 - 1e-6].sort_values("total_mm").head(3)).loc["shape"]] \
            if math.isfinite(n9) else "--"
    for s in PROFILES:
        rr = sens[sens["shape"] == s].iloc[0]
        m[f"TLSENS{PKEY[s]}C"] = rf"\SI{{{rr.c_mm:.0f}}}{{mm}}"
        m[f"TLSENS{PKEY[s]}D"] = rf"\SI{{{rr.d9km:+.0f}}}{{m}}"
    text = "% generated by studies/total_length.py -- do not edit\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())
    (secdir / "tl_macros.tex").write_text(text, encoding="utf-8")
    return m


# ------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--report", default="report_650", help="report folder for figures/ and sections/")
    ap.add_argument("--cad-copy", default="CFD/geometry_650", help="folder that also receives the STEP files")
    ap.add_argument("--replot", action="store_true", help="only redraw the figures and tables from saved results")
    a = ap.parse_args()
    mdl = Model(Path(a.config))
    out = Path(mdl.cfg.project.output_dir) / "total_length"
    out.mkdir(parents=True, exist_ok=True)
    rep = ROOT / a.report
    tgt = mdl.cfg.flight.target_apogee
    if a.replot:
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        d = flag(pd.read_csv(out / "design_points.csv").drop(columns=["d9km", "best", "design"]), tgt)
        ts = pd.read_csv(out / "total_length_sweep.csv")
        figures(mdl, pd.read_csv(out / "coupler_sweep.csv"), ts, d, rep / "figures")
        write_latex(mdl, d, ts, pd.read_csv(out / "cfd_check.csv"), s["loads"], s["heating"], s["cad"],
                    pd.read_csv(out / "sensitivity_3u.csv"), s["dz_dm"], rep / "sections")
        print(f"figures and tables redrawn in {rep}")
        return
    b_shape, b_f = mdl.st.baseline
    print(f"k = {mdl.st.k:.4f}; baseline apogee {mdl.apogee(b_shape, 2 * mdl.R * b_f, 0.0):.0f} m")

    sw = sweep(mdl)
    sw.to_csv(out / "coupler_sweep.csv", index=False)
    ts = total_sweep(mdl)
    ts.to_csv(out / "total_length_sweep.csv", index=False)
    d = design_points(mdl)
    d.to_csv(out / "design_points.csv", index=False)
    cfdc = cfd_check(mdl)
    cfdc.to_csv(out / "cfd_check.csv", index=False)
    dz_dm = float(mdl.st.sensitivities()["dz_dm"])
    # 3U sensitivity (study A): 3U CubeSat envelope (340.5 mm) instead of the GRV requirement
    sens = []
    for shape in PROFILES:
        c = mdl.coupler_for(shape, RULES["inside"], l_grv=0.3405 + mdl.cfg.payload.axial_clearance)
        apo = mdl.apogee(shape, TOTAL - c, c)
        sens.append(dict(shape=shape, c_mm=1e3 * c, apogee=apo, d9km=apo - tgt))
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity_3u.csv", index=False)

    loads, heat = {}, {}
    for _, r in d.iterrows():
        s, L, c = r["shape"], 1e-3 * r.L_mm, 1e-3 * r.c_mm
        m_pay = CANSAT_MASS + (GRV_MASS if r.rule == "inside" else 0.0)
        nom = mdl.trajectory(s, L, c, detail=True)
        env = mdl.trajectory(s, L, c, envelope=True, detail=True)
        cases = {"nominal": mdl.joint_loads(s, L, c, nom, False, m_pay),
                 "envelope": mdl.joint_loads(s, L, c, env, True, m_pay)}
        worst = max(cases, key=lambda k: cases[k]["F_max"])
        ld = dict(cases[worst], case=worst)
        for key in ("q_max", "bend_per_deg", "D_nose_max"):
            ld[key] = max(v[key] for v in cases.values())
        ld["F_min"] = min(v["F_min"] for v in cases.values())
        ld["M_max"] = [float(nom["M"].max()), float(env["M"].max())]
        loads[f"{r.rule}:{s}"] = ld
        heat[f"{r.rule}:{s}"] = mdl.heating(s, L, nom, env)
    cad = export_cad(mdl, d, out / "cad")
    if a.cad_copy:
        dst = ROOT / a.cad_copy
        dst.mkdir(parents=True, exist_ok=True)
        for cc in cad.values():
            for ext in ("step", "csv"):
                src = out / "cad" / f"{cc['stem']}.{ext}"
                if src.exists():
                    shutil.copy2(src, dst / src.name)

    figures(mdl, sw, ts, d, rep / "figures")
    macros = write_latex(mdl, d, ts, cfdc, loads, heat, cad, sens, dz_dm, rep / "sections")
    summary = dict(k=mdl.st.k, dz_dm=dz_dm, design_points=d.to_dict("records"), loads=loads, heating=heat, cad=cad,
                   sensitivity_3u=sens.to_dict("records"), cfd_check=cfdc.to_dict("records"),
                   nine_km={ru: nine_km(ts, ru, tgt) for ru in RULES})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print(json.dumps(dict(loads=loads, heating=heat, nine_km=summary["nine_km"], dz_dm=dz_dm), indent=1, default=float))
    print(sens.round(1).to_string(index=False))
    print(f"{len(macros)} macros; outputs in {out} and {rep}")


if __name__ == "__main__":
    main()

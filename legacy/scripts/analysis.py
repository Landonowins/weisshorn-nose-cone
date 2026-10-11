"""Trade study: Euler wave drag -> flight performance, stability, mass, payload fit.

Reads results/euler_sweep.csv and writes results/trade_table.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from geometry import SHAPES, volume_and_area, length_with_radius, base_angle_deg
from flight import DragModel, simulate, LVO, ASSUMED

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

R = LVO["D"] / 2
BASELINE = ("vonkarman", 4.0)          # LVS-scaled length: 720 mm = 4 calibres
M_ONSET = 0.80                        # assumed onset of nose pressure-drag rise
AREAL_DENSITY = 0.98                   # kg/m2, Firehorn as-built nose (500 g; see nose_areal_density)
M_TIP_AL = 0.060                      # kg, machined Al tip insert (same for all shapes)
R_3U = 0.077                          # outer radius needed to house a 3U section (see report)
L_3U = 0.3405


def nose_areal_density():
    """Firehorn nose: x^1/2, R = 120 mm, L = 960 mm (+30 mm cylinder), 500 g measured (IR)."""
    _, S = volume_and_area("power050", 0.96, 0.12)
    S += 2 * np.pi * 0.12 * 0.03
    return 0.500 / S


def collect_npz():
    """Rebuild the results table from the saved wall-pressure files.

    Every finished Mach point writes results/cp/<study>_<shape>_f<f>_rn<rn>_<grid>_M<M>.npz,
    so points of interrupted jobs are kept. The grid is regenerated (deterministic) to get the
    exact wall-face normals, and C_D,w = -2 sum cp n_x l r over the nose faces (R = 1).
    """
    import glob
    import re
    import euler_axi as e
    from run_sweep import GRIDS
    rows, meshes = [], {}
    pat = re.compile(r"(\w+?)_(\w+?)_f([\d.]+)_rn([\d.]+)_(\w+)_M([\d.]+)\.npz$")
    for fn in sorted(glob.glob(os.path.join(RES, "cp", "*.npz"))):
        m = pat.search(os.path.basename(fn))
        if not m:
            continue
        study, shape, f, rn, grid, M = m.groups()
        f, rn, M = float(f), float(rn), float(M)
        key = (shape, f, rn, grid)
        if key not in meshes:
            ni, nc, nj, dy = GRIDS[grid]
            X, Y, ish = e.make_grid(shape, f, rn=rn, ni_nose=ni, ni_cyl=nc, nj=nj, wall_dy=dy)
            meshes[key] = e.Mesh(X, Y, ish)
        mesh = meshes[key]
        cp = np.load(fn)["cp"]
        sl = slice(0, mesh.i_sh)
        cd = -2 * np.sum(cp[sl] * mesh.njx[sl, 0] * mesh.lj[sl, 0] * mesh.rj[sl, 0])
        rows.append(dict(study=study, shape=shape, fineness=f, rn=rn, grid=grid, mach=M, cd=cd))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "euler_results.csv"), index=False)
    return df


def load_sweep(study="main"):
    """Euler results of one study; 'main' also includes the direct L/D = 4.5 runs (study 'f45')."""
    df = pd.read_csv(os.path.join(RES, "euler_results.csv"))
    studies = ["main", "f45"] if study == "main" else [study]
    d = df[df.study.isin(studies) & (df.fineness <= 5.0 + 1e-9)]
    if study == "main":
        d = d[d.rn == 0]
    return d


class WaveDrag:
    """Cd_wave(M) for one shape at arbitrary fineness.

    Supersonic: Euler values, interpolated linearly in Mach and in f (on Cd f^2,
    which is nearly constant for slender bodies). Below M = 1.1: smooth rise from 0
    at M_ONSET to the M = 1.1 value at M = 1.0, then flat (assumption; transonic
    Euler/RANS needed to refine).
    """

    MACHS = np.array([1.1, 1.2, 1.4, 1.6])

    def __init__(self, df, shape, f):
        d = df[(df["shape"] == shape) & (df.rn == 0) & (df.grid == "medium")].sort_values(["fineness", "mach"])
        # only fineness values with a usable Mach curve; with a single one, C_D f^2 = const
        counts = d.groupby("fineness").mach.nunique()
        fs = np.sort(counts[counts >= 2].index.values)
        if len(fs) == 0:
            raise ValueError(f"no Euler data for {shape}")
        d = d[d.fineness.isin(fs)]
        self.n_fineness = len(fs)
        self.machs = self.MACHS
        # each fineness on the common Mach grid (cone/ogive also have M = 1.3)
        tab = np.array([np.interp(self.machs, d[d.fineness == ff].mach, d[d.fineness == ff].cd) * ff**2
                        for ff in fs])
        w = np.array([np.interp(f, fs, tab[:, k]) for k in range(len(self.machs))])
        self.cd_sup = w / f**2

    def __call__(self, M):
        M = np.asarray(M, dtype=float)
        c11 = np.interp(1.1, self.machs, self.cd_sup)
        s = np.clip((M - M_ONSET) / (1.0 - M_ONSET), 0, 1)
        trans = c11 * s * s * (3 - 2 * s)
        sup = np.interp(M, self.machs, self.cd_sup)
        return np.where(M >= 1.1, sup, trans)


class Vehicle(DragModel):
    """Drag build-up with the non-nose drag scaled by k (calibrated to the LVO)."""
    k = 1.0

    def cd(self, M, Re, burning):
        c = self.components(M, Re, burning)
        return self.k * (sum(c.values()) - c["nose_wave"]) + c["nose_wave"]


def nose_mass(shape, L):
    _, S = volume_and_area(shape, L, R)
    return AREAL_DENSITY * S + M_TIP_AL


def make_vehicle(df, shape, f, k):
    L = 2 * R * f
    _, S = volume_and_area(shape, L, R)
    v = Vehicle(WaveDrag(df, shape, f), S, L)
    v.k = k
    return v


def calibrate(df, mass_model="isp"):
    """Scale factor on the non-nose drag so that the baseline reaches the 9 km LVO apogee."""
    def err(k):
        return simulate(make_vehicle(df, *BASELINE, k), mass_model=mass_model)["apogee"] - LVO["apogee_target"]
    return brentq(err, 0.3, 1.5, xtol=1e-4)


# ---------------------------------------------------------------- stability
def fin_cna_xcp(a=ASSUMED, d=LVO["D"]):
    """Barrowman normal-force slope and CP (from the fin root leading edge)."""
    cr, ct, s, xt, n = a["fin_root"], a["fin_tip"], a["fin_span"], a["fin_sweep"], a["n_fins"]
    lm = np.hypot(s, xt + ct / 2 - cr / 2)
    cna = (1 + (d / 2) / (s + d / 2)) * (4 * n * (s / d) ** 2) / (1 + np.sqrt(1 + (2 * lm / (cr + ct)) ** 2))
    xf = xt * (cr + 2 * ct) / (3 * (cr + ct)) + (cr + ct - cr * ct / (cr + ct)) / 6
    return cna, xf


def static_margin_shift(shape, f):
    """Change in subsonic static margin [cal] relative to the baseline nose.

    Positions are measured from the aft end so that a longer nose does not shift the
    body; the CG moves with the extra nose mass (nominal burn-out mass).
    """
    cna_f, xf = fin_cna_xcp()
    L_body = LVO["L_total"] - ASSUMED["L_nose_ref"]
    x_f_aft = ASSUMED["fin_root"] - xf                  # fins flush with the aft end

    def cp_cg(sh, ff):
        L = 2 * R * ff
        V, _ = volume_and_area(sh, L, R)
        x_n_aft = L_body + V / (np.pi * R**2)           # nose CP: L - V/A from the tip
        x_cp = (2 * x_n_aft + cna_f * x_f_aft) / (2 + cna_f)
        m = nose_mass(sh, L)
        x_m = L_body + 0.35 * L                         # nose-shell CG ~ 0.35 L above the shoulder
        return x_cp, m, x_m

    cp0, m0, xm0 = cp_cg(*BASELINE)
    cp1, m1, xm1 = cp_cg(shape, f)
    m_bo = LVO["m_wet"] - LVO["impulse"] / (LVO["isp"] * 9.80665)
    dx_cg = (m1 * xm1 - m0 * xm0) / (m_bo + m1 - m0)
    # static margin = (x_cg - x_cp)/D measured from the aft end (CG ahead of CP)
    return ((dx_cg) - (cp1 - cp0)) / LVO["D"]


L_SHOULDER = 0.060    # bonded coupler shoulder (Firehorn: 30 mm bond, longer for concentricity)
CLEARANCE_3U = 0.010  # axial margin for the payload bulkhead


def nose_bay_length(shape, L):
    """Curved length + the cylinder the 3U still needs behind the shoulder (>= the coupler)."""
    l_in = length_with_radius(shape, L, R, R_3U)
    return L + max(L_SHOULDER, L_3U + CLEARANCE_3U - l_in)


def export_profile(shape, f, rn_over_R, path, n=200):
    """CSV of the final outer mould line: x from the (blunted) tip, r, in mm."""
    from geometry import surface
    L = 2 * R * f
    x, y = surface(shape, L, R, rn=rn_over_R * R, n=4000)
    x0 = x[0]
    # resample with points clustered at the tip
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sq = s[-1] * np.linspace(0, 1, n) ** 1.6
    xs, ys = np.interp(sq, s, x), np.interp(sq, s, y)
    pd.DataFrame({"x_mm": 1e3 * (xs - x0), "r_mm": 1e3 * ys}).round(3).to_csv(path, index=False)
    return 1e3 * (L - x0)


def trade(df, k, fineness=(3.0, 3.5, 4.0, 4.5, 5.0)):
    rows = []
    for shape in SHAPES:
        for f in fineness:
            L = 2 * R * f
            V, S = volume_and_area(shape, L, R)
            try:
                veh = make_vehicle(df, shape, f, k)
            except ValueError:
                continue
            dm = nose_mass(shape, L) - nose_mass(*BASELINE[:1], 2 * R * BASELINE[1])
            tr = simulate(veh, m_nose_delta=dm, mass_model="isp")
            wd = veh.cdw
            rows.append(dict(shape=shape, fineness=f, L_mm=1e3 * L, V_L=1e3 * V, S_m2=S,
                             mass_g=1e3 * nose_mass(shape, L),
                             base_angle=base_angle_deg(shape, L, R),
                             xcp_over_L=1 - V / (np.pi * R**2 * L),
                             l3U_mm=1e3 * length_with_radius(shape, L, R, R_3U),
                             bay_mm=1e3 * nose_bay_length(shape, L),
                             cd_w_11=float(wd(1.1)), cd_w_12=float(wd(1.2)), cd_w_14=float(wd(1.4)),
                             cd_w_16=float(wd(1.6)),
                             apogee=tr["apogee"], mach_max=tr["M"].max(),
                             dSM_cal=static_margin_shift(shape, f)))
    t = pd.DataFrame(rows)
    base = t[(t["shape"] == BASELINE[0]) & (t.fineness == BASELINE[1])].apogee.values[0]
    t["d_apogee"] = t.apogee - base
    bay0 = t[(t["shape"] == BASELINE[0]) & (t.fineness == BASELINE[1])].bay_mm.values[0]
    t["d_bay_mm"] = t.bay_mm - bay0
    return t


if __name__ == "__main__":
    collect_npz()
    print("areal density from Firehorn IR: %.3f kg/m2" % nose_areal_density())
    df = load_sweep()
    k = calibrate(df)
    print("drag scale k = %.3f" % k)
    t = trade(df, k)
    t.to_csv(os.path.join(RES, "trade_table.csv"), index=False)
    # sensitivities around the baseline: +1 kg (dry and wet) and +0.01 on C_D (whole flight)
    veh = make_vehicle(df, *BASELINE, k)
    z0 = simulate(veh, mass_model="isp")["apogee"]
    z_m = simulate(veh, m_nose_delta=1.0, mass_model="isp")["apogee"]
    veh2 = make_vehicle(df, *BASELINE, k)
    base_cdw = veh2.cdw
    veh2.cdw = lambda M: base_cdw(M) + 0.01
    z_cd = simulate(veh2, mass_model="isp")["apogee"]
    with open(os.path.join(RES, "calibration.txt"), "w") as fh:
        fh.write(f"k = {k:.4f}\n")
    with open(os.path.join(RES, "sensitivity.txt"), "w") as fh:
        fh.write(f"dz/dm = {z_m - z0:.1f} m/kg\ndz/dCD = {(z_cd - z0) / 0.01:.0f} m per unit CD "
                 f"({z_cd - z0:.1f} m per 0.01)\n")
    print(open(os.path.join(RES, "sensitivity.txt")).read())
    pd.set_option("display.width", 220)
    print(t.round(4).to_string())

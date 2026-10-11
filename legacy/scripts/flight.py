"""Weisshorn point-mass ascent model with a component drag build-up.

Vehicle data come from the Weisshorn LVO (wiki, 2027_C_SE_LVO) and LVS
(2027_C_SE_LVS_DJF); every other number is an assumption listed in ASSUMED.
The nose pressure (wave) drag is the design variable and is passed in as a
function Cd_nose(M); everything else is common to all candidates.
"""
import numpy as np
from scipy.integrate import solve_ivp

G0, RAIR, GAM = 9.80665, 287.05, 1.4

# ---------------------------------------------------------------- vehicle (LVO)
LVO = dict(D=0.180, L_total=5.4, m_wet=75.0, m_dry=45.0, thrust=4500.0, t_burn=9.1,
           impulse=40960.0, isp=200.0, mach_max_lvo=1.4, apogee_target=9000.0)

# ------------------------------------------------------------ assumptions (ours)
ASSUMED = dict(
    L_nose_ref=0.72,        # LVS scales the Firehorn nose (960 mm curved) by 180/240
    rail_length=12.0,       # EuRoC launch rail [m]
    rail_elev_deg=84.0,     # EuRoC launch elevation [deg]
    site_alt=0.0,           # launch altitude ASL used for the atmosphere [m]
    roughness=20e-6,        # equivalent sand roughness, painted CFRP [m]
    n_fins=4, fin_root=0.30, fin_tip=0.12, fin_span=0.16, fin_sweep=0.18,
    fin_t=0.005, fin_le_frac=0.15,  # hexagonal section, LE/TE wedge = 15 % chord
    nozzle_exit_d=0.075,    # nozzle exit diameter, fills part of the base while burning
)


def atmosphere(h, T_ground=288.15):
    """Troposphere with a ground-temperature offset (hot/cold day). SI units."""
    lapse = 0.0065
    T = np.maximum(T_ground - lapse * h, 216.65)
    p = 101325.0 * (T / T_ground) ** (G0 / (RAIR * lapse))
    rho = p / (RAIR * T)
    mu = 1.458e-6 * T**1.5 / (T + 110.4)
    a = np.sqrt(GAM * RAIR * T)
    return T, p, rho, mu, a


def skin_friction(Re, M, length, k=ASSUMED["roughness"]):
    """Turbulent flat-plate Cf (Schlichting) with roughness limit and compressibility."""
    Re = np.maximum(Re, 1e4)
    cf = 0.455 / np.log10(Re) ** 2.58
    cf_rough = 0.032 * (k / length) ** 0.2
    cf = np.maximum(cf, cf_rough)
    corr = np.where(M < 1.0, 1 - 0.1 * M**2, 1 / (1 + 0.15 * M**2) ** 0.58)
    return cf * corr


class DragModel:
    """Zero-lift drag build-up, reference area pi D^2 / 4."""

    def __init__(self, cd_nose_wave, S_nose, L_nose, L_body=None, a=ASSUMED):
        self.cdw = cd_nose_wave             # callable: Mach -> Cd (pressure drag of the nose)
        self.D = LVO["D"]
        self.Aref = np.pi * self.D**2 / 4
        self.L_nose = L_nose
        self.L_body = L_body if L_body is not None else LVO["L_total"] - a["L_nose_ref"]
        self.L = self.L_nose + self.L_body
        self.S_body = S_nose + np.pi * self.D * self.L_body
        self.a = a
        cr, ct, s = a["fin_root"], a["fin_tip"], a["fin_span"]
        self.S_fin = 0.5 * (cr + ct) * s        # one side of one fin
        self.c_mac = 0.5 * (cr + ct)

    def components(self, M, Re_per_m, burning):
        a = self.a
        fB = self.L / self.D
        cf_b = skin_friction(Re_per_m * self.L, M, self.L)
        cd_fb = cf_b * (1 + 1 / (2 * fB)) * self.S_body / self.Aref
        cf_f = skin_friction(Re_per_m * self.c_mac, M, self.c_mac)
        tc = a["fin_t"] / self.c_mac
        cd_ff = cf_f * (1 + 2 * tc) * 2 * a["n_fins"] * self.S_fin / self.Aref
        # fin pressure drag: blunt-LE subsonic level, Ackeret hexagon supersonic, blended
        t, s, n = a["fin_t"], a["fin_span"], a["n_fins"]
        lw = a["fin_le_frac"] * self.c_mac
        frontal = n * t * s / self.Aref
        sub = 0.12 * frontal
        beta = np.sqrt(np.maximum(M**2 - 1, 1.2**2 - 1))
        sup = n * (t**2 / beta) * (2 / lw) * s / self.Aref + 0.1 * frontal
        w = np.clip((M - 0.8) / 0.4, 0, 1)
        w = w * w * (3 - 2 * w)
        cd_fp = (1 - w) * sub + w * sup
        # base drag (OpenRocket form); the nozzle exit fills part of the base while burning
        cb = np.where(M < 1, 0.12 + 0.13 * M**2, 0.25 / np.maximum(M, 1e-3))
        base_frac = 1 - (a["nozzle_exit_d"] / self.D) ** 2 if burning else 1.0
        cd_b = cb * base_frac
        # excrescences (rail buttons, aerocover seams): ERT-Sim K_e law on the wetted area
        Ke = np.where(M < 0.78, 0.00038,
                      np.where(M <= 1.04, -0.4501 * M**4 + 1.5954 * M**3 - 2.1062 * M**2 + 1.2288 * M - 0.26717,
                               np.maximum(0.0002 * M**2 - 0.0012 * M + 0.0018, 0.00038 * 0.5)))
        cd_e = Ke * self.S_body / self.Aref
        cd_nw = self.cdw(M)
        return dict(friction=cd_fb + cd_ff, fins=cd_fp, base=cd_b, excr=cd_e, nose_wave=cd_nw)

    def cd(self, M, Re_per_m, burning):
        return sum(self.components(M, Re_per_m, burning).values())


def simulate(drag, m_nose_delta=0.0, T_ground=288.15, mass_model="lvo", dt_out=0.05,
             thrust_scale=1.0):
    """Ascent to apogee. Returns a dict of time histories.

    mass_model 'lvo': linear 75 -> 45 kg over the burn (LVO wet/dry masses);
    'isp': propellant from I_t/(Isp g0) only (burn-out mass 54.1 kg).
    """
    T_b = LVO["t_burn"]
    F = LVO["thrust"] * thrust_scale
    m0 = LVO["m_wet"] + m_nose_delta
    if mass_model == "lvo":
        mdot = (LVO["m_wet"] - LVO["m_dry"]) / T_b
    else:
        mdot = LVO["impulse"] / (LVO["isp"] * G0) / T_b
    el = np.radians(ASSUMED["rail_elev_deg"])

    def rhs(t, y):
        x, z, vx, vz = y
        burning = t < T_b
        m = m0 - mdot * min(t, T_b)
        V = np.hypot(vx, vz)
        Tt, p, rho, mu, a = atmosphere(z + ASSUMED["site_alt"], T_ground)
        M = V / a
        cd = drag.cd(M, rho * V / mu if V > 0 else 0.0, burning)
        Dg = 0.5 * rho * V**2 * cd * drag.Aref
        # on the rail the direction is fixed; afterwards thrust/drag follow the velocity
        on_rail = (x * np.cos(el) + z * np.sin(el)) < ASSUMED["rail_length"]
        ex, ez = (np.cos(el), np.sin(el)) if (on_rail or V < 1e-6) else (vx / V, vz / V)
        Th = F if burning else 0.0
        ax = (Th - Dg) * ex / m
        az = (Th - Dg) * ez / m - G0
        if on_rail:
            acc = ax * ex + az * ez
            ax, az = max(acc, 0.0) * ex, max(acc, 0.0) * ez
        return [vx, vz, ax, az]

    apogee = lambda t, y: y[3]
    apogee.terminal, apogee.direction = True, -1
    sol = solve_ivp(rhs, [0, 120], [0, 0, 0, 0], events=apogee, max_step=0.02, rtol=1e-8, atol=1e-8,
                    dense_output=True)
    t = np.arange(0, sol.t[-1], dt_out)
    x, z, vx, vz = sol.sol(t)
    V = np.hypot(vx, vz)
    T, p, rho, mu, a = atmosphere(z + ASSUMED["site_alt"], T_ground)
    M = V / a
    return dict(t=t, z=z, x=x, V=V, M=M, T=T, p=p, rho=rho, mu=mu, q=0.5 * rho * V**2,
                apogee=sol.y[1, -1], t_apogee=sol.t[-1])

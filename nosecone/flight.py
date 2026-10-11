"""Point-mass ascent model with a component drag build-up.

The nose wave drag is the design variable and is passed in as a function C_D,w(M); everything
else (friction, fins, base, excrescences) is common to all candidates and can be scaled by a
single factor k, calibrated so that the baseline nose reaches the target apogee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import constants as C
from .config import Config

ArrayLike = Union[float, np.ndarray]


# ------------------------------------------------------------------------------ atmosphere
@dataclass(frozen=True)
class Atmosphere:
    """Troposphere with a ground-temperature offset (hot / cold day), SI units."""

    ground_temperature: float = 288.15
    ground_pressure: float = 101325.0
    lapse_rate: float = 0.0065
    tropopause_temperature: float = 216.65

    @classmethod
    def from_config(cls, cfg: Config, ground_temperature: Optional[float] = None) -> "Atmosphere":
        """Atmosphere of the configuration, optionally with another ground temperature [K]."""
        a = cfg.atmosphere
        T0 = a.ground_temperature if ground_temperature is None else ground_temperature
        return cls(T0, a.ground_pressure, a.lapse_rate, a.tropopause_temperature)

    def __call__(self, h: ArrayLike) -> tuple:
        """Temperature, pressure, density, viscosity and speed of sound at altitude h [m]."""
        T_ground, lapse = self.ground_temperature, self.lapse_rate
        T = np.maximum(T_ground - lapse * h, self.tropopause_temperature)
        p = self.ground_pressure * (T / T_ground) ** (C.G0 / (C.R_AIR * lapse))
        rho = p / (C.R_AIR * T)
        mu = C.SUTHERLAND_C1 * T**1.5 / (T + C.SUTHERLAND_S)
        a = np.sqrt(C.GAMMA * C.R_AIR * T)
        return T, p, rho, mu, a


def skin_friction(Re: ArrayLike, M: ArrayLike, length: float, roughness: float) -> np.ndarray:
    """Turbulent flat-plate C_f (Schlichting) with roughness limit and compressibility correction."""
    Re = np.maximum(Re, C.CF_RE_MIN)
    cf = C.CF_SCHLICHTING_A / np.log10(Re) ** C.CF_SCHLICHTING_B
    cf_rough = C.CF_ROUGH_A * (roughness / length) ** C.CF_ROUGH_B
    cf = np.maximum(cf, cf_rough)
    corr = np.where(M < 1.0, 1 - C.CF_COMP_SUB * M**2, 1 / (1 + C.CF_COMP_SUP_A * M**2) ** C.CF_COMP_SUP_B)
    return cf * corr


# ------------------------------------------------------------------------------ drag model
class DragModel:
    """Zero-lift drag build-up of the vehicle, reference area pi D^2 / 4.

    Args:
        cfg: configuration (vehicle, fins).
        cd_nose_wave: callable Mach -> C_D,w of the nose.
        S_nose: wetted area of the nose [m^2].
        L_nose: nose length [m].
        k: factor on everything except the nose wave drag.
    """

    def __init__(self, cfg: Config, cd_nose_wave: Callable, S_nose: float, L_nose: float, k: float = 1.0):
        v, f = cfg.vehicle, cfg.fins
        self.cdw = cd_nose_wave
        self.k = k
        self.D = v.diameter
        self.Aref = np.pi * self.D**2 / 4
        self.L_nose = L_nose
        self.L_body = v.length - v.reference_nose_length
        self.L = self.L_nose + self.L_body
        self.S_body = S_nose + np.pi * self.D * self.L_body
        self.roughness = v.surface_roughness
        self.nozzle_exit_d = v.nozzle_exit_diameter
        self.fins = f
        cr, ct, s = f.root_chord, f.tip_chord, f.span
        self.S_fin = 0.5 * (cr + ct) * s        # one side of one fin
        self.c_mac = 0.5 * (cr + ct)

    def components(self, M: ArrayLike, Re_per_m: ArrayLike, burning: bool) -> dict:
        """Drag coefficients: friction, fins (pressure), base, excr(escences), nose_wave (unscaled)."""
        fn = self.fins
        fB = self.L / self.D
        cf_b = skin_friction(Re_per_m * self.L, M, self.L, self.roughness)
        cd_fb = cf_b * (1 + 1 / (2 * fB)) * self.S_body / self.Aref
        cf_f = skin_friction(Re_per_m * self.c_mac, M, self.c_mac, self.roughness)
        tc = fn.thickness / self.c_mac
        cd_ff = cf_f * (1 + 2 * tc) * 2 * fn.count * self.S_fin / self.Aref
        # fin pressure drag: blunt-LE subsonic level, Ackeret hexagon supersonic, blended
        t, s, n = fn.thickness, fn.span, fn.count
        lw = fn.edge_fraction * self.c_mac
        frontal = n * t * s / self.Aref
        sub = C.FIN_SUBSONIC_CD * frontal
        beta = np.sqrt(np.maximum(M**2 - 1, C.FIN_ACKERET_MIN_MACH**2 - 1))
        sup = n * (t**2 / beta) * (2 / lw) * s / self.Aref + C.FIN_SUPERSONIC_BLUNT_CD * frontal
        w = np.clip((M - C.FIN_BLEND_START) / C.FIN_BLEND_WIDTH, 0, 1)
        w = w * w * (3 - 2 * w)
        cd_fp = (1 - w) * sub + w * sup
        # base drag (OpenRocket form); the nozzle exit fills part of the base while burning
        cb = np.where(M < 1, C.BASE_CD_SUB_A + C.BASE_CD_SUB_B * M**2, C.BASE_CD_SUP / np.maximum(M, 1e-3))
        base_frac = 1 - (self.nozzle_exit_d / self.D) ** 2 if burning else 1.0
        cd_b = cb * base_frac
        # excrescences (rail buttons, seams): ERT-Sim K_e law on the wetted area
        a4, a3, a2, a1, a0 = C.EXCRESCENCE_POLY_TRANSONIC
        b2, b1, b0 = C.EXCRESCENCE_POLY_SUPERSONIC
        Ke = np.where(M < C.EXCRESCENCE_MACH_A, C.EXCRESCENCE_KE_LOW,
                      np.where(M <= C.EXCRESCENCE_MACH_B, a4 * M**4 + a3 * M**3 + a2 * M**2 + a1 * M + a0,
                               np.maximum(b2 * M**2 + b1 * M + b0, C.EXCRESCENCE_KE_LOW * 0.5)))
        cd_e = Ke * self.S_body / self.Aref
        cd_nw = self.cdw(M)
        return dict(friction=cd_fb + cd_ff, fins=cd_fp, base=cd_b, excr=cd_e, nose_wave=cd_nw)

    def cd(self, M: ArrayLike, Re_per_m: ArrayLike, burning: bool) -> ArrayLike:
        """Total C_D, the non-nose part scaled by k."""
        c = self.components(M, Re_per_m, burning)
        return self.k * (sum(c.values()) - c["nose_wave"]) + c["nose_wave"]


# ------------------------------------------------------------------------------ trajectory
def burnout_mass(cfg: Config, mass_model: str) -> float:
    """Mass at burn-out [kg] for a mass model ('isp' or 'lvo')."""
    v, e = cfg.vehicle, cfg.engine
    if mass_model == "lvo":
        return v.mass_dry
    return v.mass_wet - e.total_impulse / (e.specific_impulse * C.G0)


def simulate(cfg: Config, drag: DragModel, m_nose_delta: float = 0.0, ground_temperature: Optional[float] = None,
             mass_model: Optional[str] = None, dt_out: Optional[float] = None, thrust_scale: float = 1.0) -> dict:
    """Ascent from the rail to apogee.

    Args:
        cfg: configuration (engine, vehicle masses, launch, atmosphere, flight numerics).
        drag: drag model of the vehicle.
        m_nose_delta: mass added to the vehicle [kg] (nose mass relative to the baseline).
        ground_temperature: overrides atmosphere.ground_temperature [K].
        mass_model: 'isp' (propellant from I_t / (I_sp g0)) or 'lvo' (wet -> dry); default flight.mass_model.
        dt_out: output sampling [s]; default flight.output_step.
        thrust_scale: factor on the thrust.

    Returns:
        Dictionary of time histories (t, z, x, V, M, T, p, rho, mu, q) and apogee, t_apogee.
    """
    e, v, la, fl = cfg.engine, cfg.vehicle, cfg.launch, cfg.flight
    mass_model = mass_model or fl.mass_model
    dt_out = dt_out or fl.output_step
    atm = Atmosphere.from_config(cfg, ground_temperature)
    T_b = e.burn_time
    F = e.thrust * thrust_scale
    m0 = v.mass_wet + m_nose_delta
    if mass_model == "lvo":
        mdot = (v.mass_wet - v.mass_dry) / T_b
    else:
        mdot = e.total_impulse / (e.specific_impulse * C.G0) / T_b
    el = np.radians(la.rail_elevation)

    def rhs(t, y):
        x, z, vx, vz = y
        burning = t < T_b
        m = m0 - mdot * min(t, T_b)
        V = np.hypot(vx, vz)
        Tt, p, rho, mu, a = atm(z + la.site_altitude)
        M = V / a
        cd = drag.cd(M, rho * V / mu if V > 0 else 0.0, burning)
        Dg = 0.5 * rho * V**2 * cd * drag.Aref
        # on the rail the direction is fixed; afterwards thrust and drag follow the velocity
        on_rail = (x * np.cos(el) + z * np.sin(el)) < la.rail_length
        ex, ez = (np.cos(el), np.sin(el)) if (on_rail or V < 1e-6) else (vx / V, vz / V)
        Th = F if burning else 0.0
        ax = (Th - Dg) * ex / m
        az = (Th - Dg) * ez / m - C.G0
        if on_rail:
            acc = ax * ex + az * ez
            ax, az = max(acc, 0.0) * ex, max(acc, 0.0) * ez
        return [vx, vz, ax, az]

    apogee = lambda t, y: y[3]  # noqa: E731
    apogee.terminal, apogee.direction = True, -1
    sol = solve_ivp(rhs, [0, fl.t_max], [0, 0, 0, 0], events=apogee, max_step=fl.max_step, rtol=fl.rtol,
                    atol=fl.atol, dense_output=True)
    t = np.arange(0, sol.t[-1], dt_out)
    x, z, vx, vz = sol.sol(t)
    V = np.hypot(vx, vz)
    T, p, rho, mu, a = atm(z + la.site_altitude)
    M = V / a
    return dict(t=t, z=z, x=x, V=V, M=M, T=T, p=p, rho=rho, mu=mu, q=0.5 * rho * V**2,
                apogee=sol.y[1, -1], t_apogee=sol.t[-1])


def calibrate(cfg: Config, make_drag: Callable[[float], DragModel]) -> float:
    """Factor k on the non-nose drag such that the baseline reaches flight.target_apogee.

    Args:
        cfg: configuration.
        make_drag: function k -> drag model of the baseline vehicle.
    """
    fl = cfg.flight

    def err(k):
        return simulate(cfg, make_drag(k), mass_model=fl.mass_model)["apogee"] - fl.target_apogee
    lo, hi = fl.calibration_bracket
    try:
        return brentq(err, lo, hi, xtol=fl.calibration_tolerance)
    except ValueError as exc:
        raise ValueError(f"calibration failed: the target apogee {fl.target_apogee:.0f} m is not reached for k in "
                         f"[{lo}, {hi}] (apogee {err(lo) + fl.target_apogee:.0f} m to "
                         f"{err(hi) + fl.target_apogee:.0f} m). Widen flight.calibration_bracket or check the "
                         f"vehicle data, or set flight.calibrate: false") from exc

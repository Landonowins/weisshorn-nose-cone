"""Aerodynamic heating of the nose tip and the shell along a trajectory.

Stagnation point: laminar Fay-Riddell form without dissociation,
    h_s = 0.763 Pr^-0.6 sqrt(rho_s mu_s du_e/dx) c_p,  du_e/dx = sqrt(2 (p_s - p_inf) / rho_s) / r_n
(modified-Newtonian velocity gradient). Shell: turbulent flat plate with Eckert's reference
temperature and the Reynolds analogy, St = 0.0296 Re_x*^-0.2 Pr^-2/3, driven by the recovery
temperature T_r = T_inf (1 + r (gamma - 1)/2 M^2), r = Pr^(1/3).
Wall response: 1D transient conduction (explicit finite differences), adiabatic inner face.
Metal tip insert: lumped capacitance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

from . import constants as C
from .geometry import Profile, surface

G = C.GAMMA
ArrayLike = Union[float, np.ndarray]
PR = C.PRANDTL


@dataclass(frozen=True)
class Material:
    """Thermal properties of a wall material."""

    label: str
    conductivity: float     # W/(m K)
    density: float          # kg/m^3
    specific_heat: float    # J/(kg K)

    @classmethod
    def from_dict(cls, name: str, d: Mapping) -> "Material":
        """Material from a heating.materials entry."""
        return cls(str(d.get("label", name)), float(d["conductivity"]), float(d["density"]),
                   float(d["specific_heat"]))


def sutherland(T: ArrayLike) -> ArrayLike:
    """Dynamic viscosity of air [Pa s] (Sutherland's law)."""
    return C.SUTHERLAND_C1 * T**1.5 / (T + C.SUTHERLAND_S)


def stagnation_state(M: ArrayLike, p: ArrayLike, T: ArrayLike) -> tuple:
    """Stagnation pressure behind the bow shock (isentropic if subsonic) and T_0."""
    M = np.asarray(M, dtype=float)
    T0 = T * (1 + 0.5 * (G - 1) * M**2)
    p_isen = p * (1 + 0.5 * (G - 1) * M**2) ** (G / (G - 1))
    Ms = np.maximum(M, 1.0)  # Rayleigh pitot formula for M > 1
    p_pitot = p * (((G + 1) ** 2 * Ms**2) / (4 * G * Ms**2 - 2 * (G - 1))) ** (G / (G - 1)) \
        * (1 - G + 2 * G * Ms**2) / (G + 1)
    p0 = np.where(M <= 1, p_isen, p_pitot)
    return (p0 if p0.ndim else float(p0)), T0


def h_stagnation(M: float, p: float, T: float, rn: float) -> tuple[float, float]:
    """Laminar stagnation-point heat-transfer coefficient [W/(m^2 K)] and T_0 [K]."""
    p0, T0 = stagnation_state(M, p, T)
    rho_s = p0 / (C.R_AIR * T0)
    due = np.sqrt(max(2 * (p0 - p), 0.0) / rho_s) / rn
    return C.FAY_RIDDELL_COEFF * PR**C.FAY_RIDDELL_PR_EXP * np.sqrt(rho_s * sutherland(T0) * due) * C.CP_AIR, T0


def h_turbulent(M: float, p: float, T: float, V: float, x: float, Tw: float) -> tuple[float, float]:
    """Local turbulent heat-transfer coefficient at distance x from the tip, and T_r."""
    r = PR ** C.RECOVERY_PR_EXP
    Tr = T * (1 + r * 0.5 * (G - 1) * M**2)
    Ts = T * (1 + C.ECKERT_A * M**2 + C.ECKERT_B * (Tw / T - 1))      # Eckert reference temperature
    rho_s = p / (C.R_AIR * Ts)
    Re = rho_s * V * x / sutherland(Ts)
    St = C.STANTON_COEFF * max(Re, C.HEAT_RE_MIN) ** C.STANTON_RE_EXP * PR ** C.STANTON_PR_EXP
    return St * rho_s * V * C.CP_AIR, Tr


def wall_response(traj: dict, material: Material, thickness: float, station: Union[str, float],
                  rn: float = None, n: int = 40, T_init: float = None) -> dict:
    """Outer and inner wall temperature histories of a slab heated on one face.

    Args:
        traj: trajectory (t, M, p, T, V).
        material: wall material.
        thickness: wall thickness [m].
        station: 'stagnation' (laminar, needs rn) or a distance x [m] from the tip (turbulent).
        rn: tip radius [m] for the stagnation station.
        n: number of cells through the thickness.
        T_init: initial temperature [K] (default: ambient at launch).
    """
    k, rho, c = material.conductivity, material.density, material.specific_heat
    a = k / (rho * c)
    dx = thickness / n
    t, M, p, T, V = traj["t"], traj["M"], traj["p"], traj["T"], traj["V"]
    dt_stab = C.FD_STABILITY * dx**2 / a
    Tw = np.full(n + 1, T_init if T_init is not None else T[0])
    out_o, out_i, hist_h, hist_Tr = [], [], [], []
    for kk in range(len(t) - 1):
        tt, t1 = t[kk], t[kk + 1]
        if station == "stagnation":
            h, Tr = h_stagnation(M[kk], p[kk], T[kk], rn)
        else:
            h, Tr = h_turbulent(M[kk], p[kk], T[kk], V[kk], station, Tw[0])
        # explicit FD with a Biot-limited step
        dt_h = C.FD_STABILITY * dx * rho * c * dx / (h * dx + k)
        nsub = int(np.ceil((t1 - tt) / min(dt_stab, dt_h)))
        dts = (t1 - tt) / nsub
        for _ in range(nsub):
            Tn = Tw.copy()
            Tn[1:-1] = Tw[1:-1] + a * dts / dx**2 * (Tw[2:] - 2 * Tw[1:-1] + Tw[:-2])
            Tn[0] = Tw[0] + 2 * dts / (rho * c * dx) * (h * (Tr - Tw[0]) + k * (Tw[1] - Tw[0]) / dx)
            Tn[-1] = Tw[-1] + 2 * a * dts / dx**2 * (Tw[-2] - Tw[-1])
            Tw = Tn
        out_o.append(Tw[0])
        out_i.append(Tw[-1])
        hist_h.append(h)
        hist_Tr.append(Tr)
    return dict(t=t[1:], T_out=np.array(out_o), T_in=np.array(out_i), h=np.array(hist_h), T_r=np.array(hist_Tr))


def tip_insert(traj: dict, profile: Profile, L: float, R: float, rn: float, x_tip: float, mass: float,
               material: Material) -> dict:
    """Lumped metal tip insert covering the nose from the apex to x_tip.

    Spherical cap at 0.55 h_s (hemisphere-averaged laminar value, driven by T_0); the remaining
    lateral area at the turbulent h evaluated at x_tip / 2 (driven by T_r).
    """
    c = material.specific_heat
    x, y = surface(profile, L, R, rn, n=3000)
    sel = x <= x_tip
    ds = np.hypot(np.diff(x[sel]), np.diff(y[sel]))
    A_tot = np.sum(2 * np.pi * 0.5 * (y[sel][1:] + y[sel][:-1]) * ds)
    A_cap = 2 * np.pi * rn**2 * C.TIP_CAP_AREA_FRACTION
    A_lat = A_tot - A_cap
    t, M, p, T, V = traj["t"], traj["M"], traj["p"], traj["T"], traj["V"]
    Tt = T[0]
    out = []
    for kk in range(len(t) - 1):
        hs, T0 = h_stagnation(M[kk], p[kk], T[kk], rn)
        ht, Tr = h_turbulent(M[kk], p[kk], T[kk], V[kk], 0.5 * x_tip, Tt)
        Q = C.HEMISPHERE_H_FACTOR * hs * A_cap * (T0 - Tt) + ht * A_lat * (Tr - Tt)
        Tt += (t[kk + 1] - t[kk]) * Q / (mass * c)
        out.append(Tt)
    return dict(t=t[1:], T=np.array(out), A=A_tot)

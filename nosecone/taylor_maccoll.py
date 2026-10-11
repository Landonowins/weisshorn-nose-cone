"""Exact inviscid supersonic flow over a sharp cone (Taylor-Maccoll), used to validate the solver."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import constants as C

G = C.GAMMA


def _tm_rhs(th: float, y: list[float]) -> list[float]:
    # y = (V_r, V_theta) normalised by V_max
    vr, vt = y
    a2 = 0.5 * (G - 1) * (1 - vr**2 - vt**2)
    dvt = (vr * vt**2 - a2 * (2 * vr + vt / np.tan(th))) / (a2 - vt**2)
    return [vt, dvt]


def _cone_from_shock(M: float, beta: float) -> tuple[float, float]:
    """Integrate inwards from a shock angle beta; return the cone angle and the surface V_r."""
    Mn1 = M * np.sin(beta)
    Mn2 = np.sqrt((1 + 0.5 * (G - 1) * Mn1**2) / (G * Mn1**2 - 0.5 * (G - 1)))
    delta = np.arctan(2 / np.tan(beta) * (M**2 * np.sin(beta) ** 2 - 1) / (M**2 * (G + np.cos(2 * beta)) + 2))
    M2 = Mn2 / np.sin(beta - delta)
    V2 = 1 / np.sqrt(2 / ((G - 1) * M2**2) + 1)
    vr, vt = V2 * np.cos(beta - delta), -V2 * np.sin(beta - delta)
    ev = lambda th, y: y[1]  # noqa: E731
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(_tm_rhs, [beta, C.TM_THETA_MIN], [vr, vt], events=ev, rtol=C.TM_RTOL, atol=C.TM_ATOL)
    if sol.t_events[0].size == 0:
        return np.nan, np.nan
    return sol.t_events[0][0], sol.y_events[0][0][0]


def cone_cp(M: float, theta_c_deg: float) -> tuple[float, float, float]:
    """Surface pressure coefficient, shock angle [deg] and surface Mach number of a sharp cone.

    Raises ValueError if the shock is detached (no conical solution).
    """
    tc = np.radians(theta_c_deg)
    mu = np.arcsin(1 / M)
    f = lambda b: _cone_from_shock(M, b)[0] - tc  # noqa: E731
    beta = brentq(f, mu + C.TM_BETA_OFFSET, np.radians(C.TM_BETA_MAX_DEG), xtol=C.TM_XTOL)
    _, vr = _cone_from_shock(M, beta)
    Mc = np.sqrt(2 / (G - 1) * vr**2 / (1 - vr**2))
    # total-pressure loss across the (straight) shock, isentropic afterwards
    Mn1 = M * np.sin(beta)
    p02_p01 = ((G + 1) * Mn1**2 / ((G - 1) * Mn1**2 + 2)) ** (G / (G - 1)) * \
              ((G + 1) / (2 * G * Mn1**2 - (G - 1))) ** (1 / (G - 1))
    p_p01 = p02_p01 * (1 + 0.5 * (G - 1) * Mc**2) ** (-G / (G - 1))
    pinf_p01 = (1 + 0.5 * (G - 1) * M**2) ** (-G / (G - 1))
    cp = (p_p01 / pinf_p01 - 1) / (0.5 * G * M**2)
    return cp, np.degrees(beta), Mc

"""Exact inviscid supersonic flow over a sharp cone (Taylor-Maccoll), for validation."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 1.4


def _tm_rhs(th, y):
    # y = (V_r, V_theta) normalised by V_max
    vr, vt = y
    a2 = 0.5 * (G - 1) * (1 - vr**2 - vt**2)
    dvt = (vr * vt**2 - a2 * (2 * vr + vt / np.tan(th))) / (a2 - vt**2)
    return [vt, dvt]


def _cone_from_shock(M, beta):
    """Integrate inward from a shock angle beta; return cone angle and surface V_r."""
    Mn1 = M * np.sin(beta)
    Mn2 = np.sqrt((1 + 0.5 * (G - 1) * Mn1**2) / (G * Mn1**2 - 0.5 * (G - 1)))
    delta = np.arctan(2 / np.tan(beta) * (M**2 * np.sin(beta) ** 2 - 1) / (M**2 * (G + np.cos(2 * beta)) + 2))
    M2 = Mn2 / np.sin(beta - delta)
    V2 = 1 / np.sqrt(2 / ((G - 1) * M2**2) + 1)
    vr, vt = V2 * np.cos(beta - delta), -V2 * np.sin(beta - delta)
    ev = lambda th, y: y[1]
    ev.terminal, ev.direction = True, 1
    sol = solve_ivp(_tm_rhs, [beta, 1e-3], [vr, vt], events=ev, rtol=1e-10, atol=1e-12)
    if sol.t_events[0].size == 0:
        return np.nan, np.nan
    return sol.t_events[0][0], sol.y_events[0][0][0]


def cone_cp(M, theta_c_deg):
    """Surface pressure coefficient and shock angle for a cone of half-angle theta_c."""
    tc = np.radians(theta_c_deg)
    mu = np.arcsin(1 / M)
    f = lambda b: _cone_from_shock(M, b)[0] - tc
    beta = brentq(f, mu + 1e-4, np.radians(80.0), xtol=1e-12)
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


if __name__ == "__main__":
    for M, tc in [(2.0, 15.0), (1.4, 10.0), (1.6, 15.0)]:
        cp, b, Mc = cone_cp(M, tc)
        print(f"M={M} theta_c={tc}: Cp={cp:.4f} beta={b:.2f} deg Mc={Mc:.3f}")

"""Aerodynamic heating of the nose tip and CFRP shell along a trajectory.

Stagnation point: laminar Fay-Riddell form without dissociation,
    h_s = 0.763 Pr^-0.6 sqrt(rho_s mu_s du_e/dx) c_p,  du_e/dx = sqrt(2 (p_s - p_inf)/rho_s) / r_n
(modified-Newtonian velocity gradient). Shell: turbulent flat plate with Eckert's
reference temperature and the Reynolds analogy, St = 0.0296 Re_x*^-0.2 Pr^-2/3,
driven by the recovery temperature T_r = T_inf (1 + r (gamma-1)/2 M^2), r = Pr^(1/3).
Wall response: 1D transient conduction (explicit FD) with an adiabatic inner face.
"""
import numpy as np

CP_AIR, PR, G = 1005.0, 0.71, 1.4

MATERIALS = {  # k [W/m/K], rho [kg/m3], c [J/kg/K]
    "CFRP (through-thickness)": (0.7, 1550.0, 900.0),
    "Al 6082-T6": (180.0, 2700.0, 900.0),
    "PET-CF (FDM)": (0.3, 1300.0, 1200.0),
}


def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


def stagnation_state(M, p, T):
    """Stagnation pressure behind the bow shock (or isentropic if subsonic) and T0."""
    M = np.asarray(M, dtype=float)
    T0 = T * (1 + 0.5 * (G - 1) * M**2)
    p_isen = p * (1 + 0.5 * (G - 1) * M**2) ** (G / (G - 1))
    Ms = np.maximum(M, 1.0)  # Rayleigh pitot formula for M > 1
    p_pitot = p * (((G + 1) ** 2 * Ms**2) / (4 * G * Ms**2 - 2 * (G - 1))) ** (G / (G - 1)) \
        * (1 - G + 2 * G * Ms**2) / (G + 1)
    p0 = np.where(M <= 1, p_isen, p_pitot)
    return (p0 if p0.ndim else float(p0)), T0


def h_stagnation(M, p, T, rn):
    p0, T0 = stagnation_state(M, p, T)
    rho_s = p0 / (287.05 * T0)
    due = np.sqrt(max(2 * (p0 - p), 0.0) / rho_s) / rn
    return 0.763 * PR**-0.6 * np.sqrt(rho_s * sutherland(T0) * due) * CP_AIR, T0


def h_turbulent(M, p, T, V, x, Tw):
    """Local turbulent heat-transfer coefficient at distance x from the tip."""
    r = PR ** (1 / 3)
    Tr = T * (1 + r * 0.5 * (G - 1) * M**2)
    Ts = T * (1 + 0.032 * M**2 + 0.58 * (Tw / T - 1))      # Eckert reference temperature
    rho_s = p / (287.05 * Ts)
    Re = rho_s * V * x / sutherland(Ts)
    St = 0.0296 * max(Re, 1e3) ** -0.2 * PR ** (-2 / 3)
    return St * rho_s * V * CP_AIR, Tr


def wall_response(traj, material, thickness, station, rn=None, n=40, T_init=None):
    """Outer/inner wall temperature histories for a slab heated on one face.

    station: 'stagnation' (needs rn) or a distance x [m] from the tip (turbulent).
    """
    k, rho, c = MATERIALS[material]
    a = k / (rho * c)
    dx = thickness / n
    t, M, p, T, V = traj["t"], traj["M"], traj["p"], traj["T"], traj["V"]
    dt_stab = 0.4 * dx**2 / a
    Tw = np.full(n + 1, T_init if T_init is not None else T[0])
    out_o, out_i, hist_h, hist_Tr = [], [], [], []
    for kk in range(len(t) - 1):
        tt, t1 = t[kk], t[kk + 1]
        if station == "stagnation":
            h, Tr = h_stagnation(M[kk], p[kk], T[kk], rn)
        else:
            h, Tr = h_turbulent(M[kk], p[kk], T[kk], V[kk], station, Tw[0])
        # explicit FD with a Biot-limited step
        dt_h = 0.4 * dx * rho * c * dx / (h * dx + k)
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
    return dict(t=t[1:], T_out=np.array(out_o), T_in=np.array(out_i), h=np.array(hist_h),
                T_r=np.array(hist_Tr))


def tip_insert(traj, shape, L, R, rn, x_tip, mass, material="Al 6082-T6"):
    """Lumped metal tip insert covering the nose from the apex to x_tip.

    Spherical cap at 0.55 h_s (hemisphere-averaged laminar value, driving T_0);
    remaining lateral area at the turbulent h evaluated at x_tip/2 (driving T_r).
    """
    from geometry import surface
    _, _, c = MATERIALS[material]
    x, y = surface(shape, L, R, rn, n=3000)
    sel = x <= x_tip
    ds = np.hypot(np.diff(x[sel]), np.diff(y[sel]))
    A_tot = np.sum(2 * np.pi * 0.5 * (y[sel][1:] + y[sel][:-1]) * ds)
    A_cap = 2 * np.pi * rn**2 * 0.5   # cap up to ~60 deg from the stagnation point
    A_lat = A_tot - A_cap
    t, M, p, T, V = traj["t"], traj["M"], traj["p"], traj["T"], traj["V"]
    Tt = T[0]
    out = []
    for kk in range(len(t) - 1):
        hs, T0 = h_stagnation(M[kk], p[kk], T[kk], rn)
        ht, Tr = h_turbulent(M[kk], p[kk], T[kk], V[kk], 0.5 * x_tip, Tt)
        Q = 0.55 * hs * A_cap * (T0 - Tt) + ht * A_lat * (Tr - Tt)
        Tt += (t[kk + 1] - t[kk]) * Q / (mass * c)
        out.append(Tt)
    return dict(t=t[1:], T=np.array(out), A=A_tot)


def lumped_tip(traj, rn, mass, material="Al 6082-T6", area_factor=1.0):
    """Lumped-capacitance metal tip: hemisphere at h_avg = 0.55 h_stag plus a conical skirt."""
    _, _, c = MATERIALS[material]
    A = 2 * np.pi * rn**2 * area_factor
    t, M, p, T = traj["t"], traj["M"], traj["p"], traj["T"]
    Tt = T[0]
    out = []
    for kk in range(len(t) - 1):
        h, T0 = h_stagnation(M[kk], p[kk], T[kk], rn)
        Tr = T[kk] * (1 + PR ** 0.5 * 0.5 * (G - 1) * M[kk] ** 2)
        Tt += (t[kk + 1] - t[kk]) * 0.55 * h * A * (0.5 * (T0 + Tr) - Tt) / (mass * c)
        out.append(Tt)
    return dict(t=t[1:], T=np.array(out))

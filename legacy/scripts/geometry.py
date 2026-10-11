"""Nose cone profile library for the Weisshorn nose cone trade study.

All profiles are written as y(x) with x measured from the (sharp) tip, 0 <= x <= L,
and y the local radius, y(L) = R. Optional spherical blunting replaces the tip by a
sphere cap of radius r_n that is tangent to the profile.
"""
import numpy as np
from scipy.optimize import brentq

SHAPES = {
    "cone": "Conical",
    "ogive": "Tangent ogive",
    "parabolic": "Parabolic ($K'=1$)",
    "power050": "Power series $n=1/2$",
    "power075": "Power series $n=3/4$",
    "lvhaack": "LV-Haack ($C=1/3$)",
    "vonkarman": "Von K\\'arm\\'an ($C=0$)",
}


def profile(shape, xi):
    """Dimensionless radius eta = y/R as a function of xi = x/L in [0, 1]."""
    xi = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    if shape == "cone":
        return xi
    if shape == "parabolic":
        return 2.0 * xi - xi**2
    if shape.startswith("power"):
        n = int(shape[5:]) / 100.0
        return xi**n
    if shape in ("lvhaack", "vonkarman"):
        C = 1.0 / 3.0 if shape == "lvhaack" else 0.0
        th = np.arccos(1.0 - 2.0 * xi)
        return np.sqrt(np.maximum(th - 0.5 * np.sin(2 * th) + C * np.sin(th) ** 3, 0.0)) / np.sqrt(np.pi)
    raise ValueError(shape)


def radius(shape, x, L, R):
    """Local radius y(x) of a sharp nose of length L and base radius R."""
    if shape == "ogive":
        rho = (R**2 + L**2) / (2 * R)
        x = np.clip(x, 0.0, L)
        return np.sqrt(np.maximum(rho**2 - (L - x) ** 2, 0.0)) + R - rho
    return R * profile(shape, np.asarray(x) / L)


def slope(shape, x, L, R, h=None):
    """dy/dx by central differences (one-sided at the ends)."""
    h = h or 1e-7 * L
    x = np.asarray(x, dtype=float)
    xp = np.minimum(x + h, L)
    xm = np.maximum(x - h, 0.0)
    return (radius(shape, xp, L, R) - radius(shape, xm, L, R)) / (xp - xm)


def blunt_tangent_point(shape, L, R, rn):
    """Tangency point x_t and sphere centre x_c for a tip sphere of radius rn.

    Tangency requires the profile normal at x_t to pass through (x_c, 0):
    x_c = x_t + y y'  and  rn^2 = y^2 (1 + y'^2).
    """
    if rn <= 0:
        return 0.0, 0.0
    f = lambda x: radius(shape, x, L, R) ** 2 * (1 + slope(shape, x, L, R) ** 2) - rn**2
    x_t = brentq(f, 1e-9 * L, 0.999 * L, xtol=1e-12 * L)
    y, yp = radius(shape, x_t, L, R), slope(shape, x_t, L, R)
    return x_t, x_t + y * yp


def surface(shape, L, R, rn=0.0, n=600, cluster=3.0):
    """Ordered (x, y) points from the tip (on the axis) to the shoulder (x = L).

    Points are clustered towards the tip (exponent `cluster`) because the curvature
    is concentrated there. For a blunted tip the x origin stays at the sharp-nose apex,
    so the blunted nose starts at x = x_c - rn > 0.
    """
    s = np.linspace(0.0, 1.0, n) ** cluster
    if rn <= 0:
        x = s * L
        return x, radius(shape, x, L, R)
    x_t, x_c = blunt_tangent_point(shape, L, R, rn)
    # sphere cap: angle phi from the stagnation point (phi=0) to the tangency point
    phi_t = np.arctan2(radius(shape, x_t, L, R), x_t - x_c) if x_t != x_c else np.pi / 2
    phi_t = np.pi - phi_t  # measured from the upstream axis
    n_cap = max(30, n // 5)
    phi = np.linspace(0.0, phi_t, n_cap)
    xc = x_c - rn * np.cos(phi)
    yc = rn * np.sin(phi)
    x = x_t + (L - x_t) * s[1:]
    return np.concatenate([xc, x]), np.concatenate([yc, radius(shape, x, L, R)])


def volume_and_area(shape, L, R, rn=0.0):
    """Enclosed volume and wetted (lateral) area of the nose."""
    x, y = surface(shape, L, R, rn, n=4000)
    vol = np.pi * np.trapezoid(y**2, x)
    ds = np.hypot(np.diff(x), np.diff(y))
    area = np.sum(2 * np.pi * 0.5 * (y[1:] + y[:-1]) * ds)
    return vol, area


def base_angle_deg(shape, L, R):
    """Angle between the profile and the cylinder at the shoulder (0 = tangent)."""
    return np.degrees(np.arctan(slope(shape, L * (1 - 1e-6), L, R)))


def tip_half_angle_deg(shape, L, R):
    """Semi-vertex angle at the sharp tip (90 deg means a blunt, vertical tip)."""
    x = 1e-6 * L
    return np.degrees(np.arctan(radius(shape, x, L, R) / x))


def length_with_radius(shape, L, R, r_req):
    """Axial length (from the shoulder forwards) over which y >= r_req."""
    if r_req >= R:
        return 0.0
    x0 = brentq(lambda x: radius(shape, x, L, R) - r_req, 0.0, L)
    return L - x0


if __name__ == "__main__":
    R, L = 0.09, 0.72
    for s in SHAPES:
        v, a = volume_and_area(s, L, R)
        print(f"{s:10s} V={v*1e3:6.2f} L  S={a:5.3f} m2  base={base_angle_deg(s, L, R):5.2f} deg  "
              f"tip={tip_half_angle_deg(s, L, R):5.1f} deg  xcp/L={1 - v/(np.pi*R**2*L):.3f}")

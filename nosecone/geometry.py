"""Nose-cone profiles and geometric properties.

A *family* is a mathematical profile with optional shape parameters (power series with
exponent n, Haack series with constant C, ...). A *profile* is a family with fixed parameter
values and a name (``power050`` is the power family with n = 0.5). All profiles are written as
the local radius y(x), with x measured from the sharp apex (0 <= x <= L) and y(L) = R.
Optional spherical blunting replaces the apex by a sphere of radius r_n tangent to the profile.

To add a family, decorate its dimensionless radius eta(xi) = y/R, xi = x/L, with
:func:`register_family` (see the user guide, section 7)::

    @register_family("secant_like", params={"a": 0.5}, doc="...")
    def _my_family(xi, a):
        return xi ** a * (2 - xi ** a)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Optional, Union

import numpy as np
from scipy.optimize import brentq

ArrayLike = Union[float, np.ndarray]


# ------------------------------------------------------------------------------- families
@dataclass
class Family:
    """A profile family: eta(xi, **params) or, if `radius` is given, y(x, L, R, **params)."""

    name: str
    params: dict[str, float]
    eta: Optional[Callable[..., np.ndarray]] = None
    radius: Optional[Callable[..., np.ndarray]] = None
    doc: str = ""


FAMILIES: dict[str, Family] = {}


def register_family(name: str, params: Optional[Mapping[str, float]] = None, doc: str = "",
                    dimensional: bool = False) -> Callable:
    """Decorator registering a profile family.

    Args:
        name: family name used in the configuration (``family: name``).
        params: shape parameters and their default values.
        doc: one-line description (shown by ``python -m nosecone profiles``).
        dimensional: if True the function is y(x, L, R, **params) instead of eta(xi, **params).
    """
    def deco(func: Callable) -> Callable:
        FAMILIES[name] = Family(name, dict(params or {}), None if dimensional else func,
                                func if dimensional else None, doc)
        return func
    return deco


@register_family("cone", doc="Straight generator: eta = xi")
def _cone(xi: np.ndarray) -> np.ndarray:
    return xi


@register_family("tangent_ogive", dimensional=True,
                 doc="Circular arc of radius rho = (R^2 + L^2) / 2R tangent to the body")
def _tangent_ogive(x: np.ndarray, L: float, R: float) -> np.ndarray:
    rho = (R**2 + L**2) / (2 * R)
    x = np.clip(x, 0.0, L)
    return np.sqrt(np.maximum(rho**2 - (L - x) ** 2, 0.0)) + R - rho


@register_family("parabolic", params={"K": 1.0},
                 doc="Parabolic series: eta = (2 xi - K xi^2) / (2 - K), 0 <= K <= 1 (K = 1: tangent)")
def _parabolic(xi: np.ndarray, K: float) -> np.ndarray:
    return (2.0 * xi - K * xi**2) / (2.0 - K)


@register_family("power", params={"n": 0.5}, doc="Power series: eta = xi^n, 0 < n <= 1 (n = 1: cone)")
def _power(xi: np.ndarray, n: float) -> np.ndarray:
    return xi**n


@register_family("haack", params={"C": 0.0},
                 doc="Haack series: eta = sqrt(theta - sin(2 theta)/2 + C sin^3 theta)/sqrt(pi), "
                     "theta = arccos(1 - 2 xi); C = 0 Von Karman, C = 1/3 LV-Haack")
def _haack(xi: np.ndarray, C: float) -> np.ndarray:
    th = np.arccos(1.0 - 2.0 * xi)
    return np.sqrt(np.maximum(th - 0.5 * np.sin(2 * th) + C * np.sin(th) ** 3, 0.0)) / np.sqrt(np.pi)


@register_family("elliptical", doc="Quarter ellipse: eta = sqrt(1 - (1 - xi)^2) (blunt, subsonic shape)")
def _elliptical(xi: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(1.0 - (1.0 - xi) ** 2, 0.0))


# ------------------------------------------------------------------------------- profiles
@dataclass(frozen=True)
class Profile:
    """A named family with fixed parameters and its plotting labels."""

    name: str
    family: str
    params: tuple = field(default_factory=tuple)   # ((name, value), ...) so that it is hashable
    label: str = ""
    long_label: str = ""

    @property
    def p(self) -> dict[str, float]:
        """Shape parameters as a dictionary."""
        return dict(self.params)

    def __str__(self) -> str:
        return self.name


BUILTIN: dict[str, Profile] = {
    "cone": Profile("cone", "cone", (), "Cone", "Conical"),
    "ogive": Profile("ogive", "tangent_ogive", (), "Tangent ogive", "Tangent ogive"),
    "parabolic": Profile("parabolic", "parabolic", (("K", 1.0),), "Parabolic", "Parabolic ($K'=1$)"),
    "power050": Profile("power050", "power", (("n", 0.5),), r"Power $n=\tfrac12$", "Power series $n=1/2$"),
    "power075": Profile("power075", "power", (("n", 0.75),), r"Power $n=\tfrac34$", "Power series $n=3/4$"),
    "lvhaack": Profile("lvhaack", "haack", (("C", 1.0 / 3.0),), "LV-Haack", "LV-Haack ($C=1/3$)"),
    "vonkarman": Profile("vonkarman", "haack", (("C", 0.0),), r"Von K\'arm\'an", "Von K\\'arm\\'an ($C=0$)"),
}
"""Built-in profiles of the Weisshorn study (their names are those of the CFD database)."""


def make_profiles(custom: Optional[Mapping[str, Mapping]] = None) -> dict[str, Profile]:
    """Built-in profiles plus the custom ones of the configuration (``nose.custom_profiles``).

    A custom entry is ``name: {family: <family>, <param>: <value>, label: <text>}``.
    """
    out = dict(BUILTIN)
    for name, spec in (custom or {}).items():
        spec = dict(spec)
        fam = spec.pop("family", None)
        if fam not in FAMILIES:
            raise ValueError(f"profile '{name}': unknown family {fam!r} (known: {sorted(FAMILIES)})")
        label = str(spec.pop("label", name))
        defaults = FAMILIES[fam].params
        unknown = set(spec) - set(defaults)
        if unknown:
            raise ValueError(f"profile '{name}': family '{fam}' has no parameter(s) {sorted(unknown)}")
        params = tuple((k, float(spec.get(k, v))) for k, v in defaults.items())
        out[name] = Profile(name, fam, params, label, label)
    return out


def profile_names(custom: Optional[Mapping[str, Mapping]] = None) -> list[str]:
    """Names of all available profiles."""
    return list(make_profiles(custom))


def as_profile(p: Union[Profile, str]) -> Profile:
    """Accept a Profile or the name of a built-in profile."""
    if isinstance(p, Profile):
        return p
    if p in BUILTIN:
        return BUILTIN[p]
    raise KeyError(f"unknown profile {p!r}; custom profiles must be passed as Profile objects")


# ------------------------------------------------------------------------- radius and slope
def eta(profile: Union[Profile, str], xi: ArrayLike) -> np.ndarray:
    """Dimensionless radius eta = y/R as a function of xi = x/L in [0, 1]."""
    pr = as_profile(profile)
    fam = FAMILIES[pr.family]
    xi = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    if fam.eta is None:
        return fam.radius(xi, 1.0, 1.0, **pr.p)
    return fam.eta(xi, **pr.p)


def radius(profile: Union[Profile, str], x: ArrayLike, L: float, R: float) -> np.ndarray:
    """Local radius y(x) of a sharp nose of length L and base radius R."""
    pr = as_profile(profile)
    fam = FAMILIES[pr.family]
    if fam.radius is not None:
        return fam.radius(x, L, R, **pr.p)
    return R * eta(pr, np.asarray(x) / L)


def slope(profile: Union[Profile, str], x: ArrayLike, L: float, R: float, h: Optional[float] = None) -> np.ndarray:
    """dy/dx by central differences (one-sided at the ends)."""
    h = h or 1e-7 * L
    x = np.asarray(x, dtype=float)
    xp = np.minimum(x + h, L)
    xm = np.maximum(x - h, 0.0)
    return (radius(profile, xp, L, R) - radius(profile, xm, L, R)) / (xp - xm)


def blunt_tangent_point(profile: Union[Profile, str], L: float, R: float, rn: float) -> tuple[float, float]:
    """Tangency point x_t and sphere centre x_c of a spherical tip of radius rn.

    Tangency requires the profile normal at x_t to pass through (x_c, 0):
    x_c = x_t + y y' and rn^2 = y^2 (1 + y'^2).
    """
    if rn <= 0:
        return 0.0, 0.0
    f = lambda x: radius(profile, x, L, R) ** 2 * (1 + slope(profile, x, L, R) ** 2) - rn**2  # noqa: E731
    x_t = brentq(f, 1e-9 * L, 0.999 * L, xtol=1e-12 * L)
    y, yp = radius(profile, x_t, L, R), slope(profile, x_t, L, R)
    return x_t, x_t + y * yp


def surface(profile: Union[Profile, str], L: float, R: float, rn: float = 0.0, n: int = 600,
            cluster: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Ordered (x, y) points from the tip (on the axis) to the shoulder (x = L).

    Points are clustered towards the tip (exponent `cluster`). For a blunted tip the x origin
    stays at the sharp apex, so the blunted nose starts at x = x_c - rn > 0.
    """
    s = np.linspace(0.0, 1.0, n) ** cluster
    if rn <= 0:
        x = s * L
        return x, radius(profile, x, L, R)
    x_t, x_c = blunt_tangent_point(profile, L, R, rn)
    # sphere cap: angle phi from the stagnation point (phi = 0) to the tangency point
    phi_t = np.arctan2(radius(profile, x_t, L, R), x_t - x_c) if x_t != x_c else np.pi / 2
    phi_t = np.pi - phi_t  # measured from the upstream axis
    n_cap = max(30, n // 5)
    phi = np.linspace(0.0, phi_t, n_cap)
    xc = x_c - rn * np.cos(phi)
    yc = rn * np.sin(phi)
    x = x_t + (L - x_t) * s[1:]
    return np.concatenate([xc, x]), np.concatenate([yc, radius(profile, x, L, R)])


@lru_cache(maxsize=4096)
def _vol_area(profile: Profile, L: float, R: float, rn: float) -> tuple[float, float]:
    x, y = surface(profile, L, R, rn, n=4000)
    vol = np.pi * np.trapezoid(y**2, x)
    ds = np.hypot(np.diff(x), np.diff(y))
    area = np.sum(2 * np.pi * 0.5 * (y[1:] + y[:-1]) * ds)
    return vol, area


def volume_and_area(profile: Union[Profile, str], L: float, R: float, rn: float = 0.0) -> tuple[float, float]:
    """Enclosed volume [m^3] and wetted (lateral) area [m^2] of the nose."""
    return _vol_area(as_profile(profile), float(L), float(R), float(rn))


def base_angle_deg(profile: Union[Profile, str], L: float, R: float) -> float:
    """Angle between the profile and the cylinder at the shoulder [deg] (0 = tangent)."""
    return np.degrees(np.arctan(slope(profile, L * (1 - 1e-6), L, R)))


def tip_half_angle_deg(profile: Union[Profile, str], L: float, R: float) -> float:
    """Semi-vertex angle at the sharp tip [deg] (90 means a blunt, vertical tip)."""
    x = 1e-6 * L
    return np.degrees(np.arctan(radius(profile, x, L, R) / x))


def length_with_radius(profile: Union[Profile, str], L: float, R: float, r_req: float) -> float:
    """Axial length (from the shoulder forwards) over which y >= r_req [m]."""
    if r_req >= R:
        return 0.0
    x0 = brentq(lambda x: radius(profile, x, L, R) - r_req, 0.0, L)
    return L - x0


def barrowman_xcp_over_L(profile: Union[Profile, str], L: float, R: float) -> float:
    """Barrowman / slender-body nose centre of pressure from the apex, x_cp/L = 1 - V/(A_b L)."""
    V, _ = volume_and_area(profile, L, R)
    return 1 - V / (np.pi * R**2 * L)


def describe_families() -> list[tuple[str, dict, str]]:
    """(name, parameters, description) of every registered family."""
    return [(f.name, f.params, f.doc) for f in FAMILIES.values()]


def iter_profiles(names: Iterable[str], profiles: Mapping[str, Profile]) -> list[Profile]:
    """Profiles for a list of names."""
    return [profiles[n] for n in names]

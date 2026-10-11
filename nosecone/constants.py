"""Physical constants and empirical model coefficients.

Everything here is fixed physics or a coefficient of a published correlation, never a design
input: design inputs live in the YAML configuration (see :mod:`nosecone.config`). Each value
states where it is used. Changing a value here changes the model for every configuration.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- air and gravity
G0: float = 9.80665
"""Standard gravity [m/s^2] (trajectory, propellant mass from I_t / (I_sp g0))."""

R_AIR: float = 287.05
"""Specific gas constant of dry air [J/(kg K)] (atmosphere, heating)."""

GAMMA: float = 1.4
"""Ratio of specific heats of air [-] (atmosphere, Euler solver, heating, Taylor-Maccoll)."""

CP_AIR: float = 1005.0
"""Specific heat of air at constant pressure [J/(kg K)] (heating)."""

PRANDTL: float = 0.71
"""Prandtl number of air [-] (heating)."""

SUTHERLAND_C1: float = 1.458e-6
"""Sutherland's law coefficient [kg/(m s K^0.5)]: mu = C1 T^1.5 / (T + S)."""

SUTHERLAND_S: float = 110.4
"""Sutherland temperature [K]."""

# ------------------------------------------------- skin friction (flight.skin_friction)
CF_SCHLICHTING_A: float = 0.455
"""Schlichting turbulent flat plate: C_f = A / (log10 Re)^B."""
CF_SCHLICHTING_B: float = 2.58
CF_ROUGH_A: float = 0.032
"""Roughness-limited flat plate: C_f = A (k / l)^B (k: sand roughness, l: length)."""
CF_ROUGH_B: float = 0.2
CF_RE_MIN: float = 1e4
"""Lower clip of the Reynolds number in the friction law."""
CF_COMP_SUB: float = 0.1
"""Compressibility correction, subsonic: C_f (1 - 0.1 M^2)."""
CF_COMP_SUP_A: float = 0.15
"""Compressibility correction, supersonic: C_f / (1 + 0.15 M^2)^0.58."""
CF_COMP_SUP_B: float = 0.58

# ------------------------------------------------------ drag build-up (flight.DragModel)
FIN_SUBSONIC_CD: float = 0.12
"""Pressure drag of blunt fin leading edges below Mach 1, per unit fin frontal area."""
FIN_SUPERSONIC_BLUNT_CD: float = 0.1
"""Residual blunt-edge drag added to the Ackeret hexagon wave drag above Mach 1."""
FIN_ACKERET_MIN_MACH: float = 1.2
"""Ackeret's beta = sqrt(M^2 - 1) is evaluated at least at this Mach number (avoids M -> 1)."""
FIN_BLEND_START: float = 0.8
"""Fin pressure drag blends from the subsonic to the supersonic law between these Mach numbers."""
FIN_BLEND_WIDTH: float = 0.4
BASE_CD_SUB_A: float = 0.12
"""OpenRocket base drag, subsonic: 0.12 + 0.13 M^2."""
BASE_CD_SUB_B: float = 0.13
BASE_CD_SUP: float = 0.25
"""OpenRocket base drag, supersonic: 0.25 / M."""
EXCRESCENCE_KE_LOW: float = 0.00038
"""ERT-Sim excrescence coefficient K_e below M = 0.78 (applied to the wetted area)."""
EXCRESCENCE_MACH_A: float = 0.78
EXCRESCENCE_MACH_B: float = 1.04
EXCRESCENCE_POLY_TRANSONIC: tuple[float, ...] = (-0.4501, 1.5954, -2.1062, 1.2288, -0.26717)
"""K_e(M) for 0.78 <= M <= 1.04: coefficients of M^4 ... M^0."""
EXCRESCENCE_POLY_SUPERSONIC: tuple[float, ...] = (0.0002, -0.0012, 0.0018)
"""K_e(M) for M > 1.04: coefficients of M^2, M^1, M^0 (floored at half the low-speed value)."""

# ----------------------------------------------------------------- heating (heating.py)
FAY_RIDDELL_COEFF: float = 0.763
"""Fay-Riddell laminar stagnation point: h_s = 0.763 Pr^-0.6 sqrt(rho_s mu_s du_e/dx) c_p."""
FAY_RIDDELL_PR_EXP: float = -0.6
ECKERT_A: float = 0.032
"""Eckert reference temperature: T* = T (1 + 0.032 M^2 + 0.58 (T_w / T - 1))."""
ECKERT_B: float = 0.58
STANTON_COEFF: float = 0.0296
"""Turbulent flat plate (Reynolds analogy): St = 0.0296 Re*^-0.2 Pr^-2/3."""
STANTON_RE_EXP: float = -0.2
STANTON_PR_EXP: float = -2 / 3
HEAT_RE_MIN: float = 1e3
"""Lower clip of the Reynolds number in the turbulent heating law."""
RECOVERY_PR_EXP: float = 1 / 3
"""Turbulent recovery factor r = Pr^(1/3)."""
HEMISPHERE_H_FACTOR: float = 0.55
"""Hemisphere-averaged laminar coefficient as a fraction of the stagnation value."""
TIP_CAP_AREA_FRACTION: float = 0.5
"""Spherical cap treated at the stagnation coefficient: 0.5 x (2 pi r_n^2), i.e. ~60 deg."""
FD_STABILITY: float = 0.4
"""Explicit conduction scheme: time step = 0.4 x stability limit (Fourier and Biot)."""

# ---------------------------------------------------------- Taylor-Maccoll (validation)
TM_RTOL: float = 1e-10
TM_ATOL: float = 1e-12
TM_THETA_MIN: float = 1e-3
"""Integration of the Taylor-Maccoll ODE stops at this ray angle [rad] if no cone is found."""
TM_BETA_MAX_DEG: float = 80.0
"""Upper bound of the shock-angle search [deg]."""
TM_BETA_OFFSET: float = 1e-4
"""Lower bound of the shock-angle search: Mach angle + this offset [rad]."""
TM_XTOL: float = 1e-12

# ------------------------------------------------- CFD grid generator (euler.make_grid)
GRID_TIP_CLUSTER: float = 4.0
"""Clustering strength of the body points towards the tip (two-sided tanh stretching)."""
GRID_SHOULDER_CLUSTER: float = 2.0
"""Clustering strength of the body points towards the shoulder."""
GRID_CYL_CLUSTER: float = 2.5
"""Clustering of the cylinder points towards the shoulder."""
GRID_UPSTREAM_SLOPE: float = 0.3
"""The outer boundary stands upstream_distance x (1 + 0.3 L/D) ahead of the tip."""
GRID_HYPERBOLA_B: float = 2.0
"""Shape parameter of the hyperbolic outer boundary [R]."""
GRID_OUTER_MARGIN: float = 2.0
"""Extra radial extent of the outer boundary at the outflow plane [R]."""
GRID_OUTER_SAMPLES: int = 4000
"""Samples used to describe the outer boundary before resampling."""
GRID_OUTER_EXPONENT: float = 0.6
"""Mapping of the body arc length to the outer boundary (less clustered than the body)."""
GRID_NORMAL_BLEND: float = 0.08
"""Grid lines leave the wall along the normal and bend towards the outer point over ~this fraction."""
GRID_BODY_SAMPLES: int = 3000
"""Samples of the body contour before resampling."""
GRID_SMOOTH_WEIGHT: float = 0.3
"""Weight of the optional Laplacian smoothing (off by default)."""

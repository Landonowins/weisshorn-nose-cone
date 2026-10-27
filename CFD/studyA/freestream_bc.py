"""Free-stream and boundary-condition values of the Fluent study A runs (LV-Haack, L = 561 mm).

Recomputes, from the trajectory Mach number, static pressure and static temperature, everything
the Fluent setup and its checks use: density, speed of sound, velocity, viscosity, dynamic
pressure, Reynolds numbers, stagnation values, the k and omega of the turbulence boundaries, the
first-cell height for y+ = 1 and the drag coefficients from the wall forces Fluent reported.

The air is Fluent's default (MW 28.966, cp 1006.43 J/(kg K), three-coefficient Sutherland), not
nosecone/constants.py, so the numbers match what Fluent used to 5-6 digits.

Usage:
    python CFD/studyA/freestream_bc.py
"""
from __future__ import annotations

import math

R_UNIVERSAL = 8314.462618          # J/(kmol K)
MW_AIR = 28.966                    # kg/kmol, Fluent default
CP_AIR = 1006.43                   # J/(kg K), Fluent default
R_AIR = R_UNIVERSAL / MW_AIR       # 287.04 J/(kg K)
GAMMA = CP_AIR / (CP_AIR - R_AIR)  # 1.399
MU0, T0_SUTH, S_SUTH = 1.716e-5, 273.11, 110.56   # Sutherland, three coefficients

D_REF = 0.18                       # body diameter [m]
A_REF = 0.02545                    # reference area entered in Fluent, pi D^2 / 4 rounded [m2]
X_SHOULDER = 0.55865               # tip to shoulder [m], from the STEP file
TURB_INTENSITY = 0.01              # inlet, farfield and outlet
TURB_VISC_RATIO = 10.0

# (Mach, altitude [m], p [Pa], T [K], wall_nose pressure force [N], wall_nose viscous force [N])
CASES = [
    (0.95, 928, 90666.0, 282.1, 6.1726939, 36.151514),
    (1.20, 1478, 84789.0, 278.5, 206.08883, 48.664626),
    (1.47, 2189, 77649.0, 273.9, 287.72219, 61.693027),
]


def sutherland(t: float) -> float:
    """Dynamic viscosity of air [Pa s] at temperature t [K]."""
    return MU0 * (t / T0_SUTH) ** 1.5 * (T0_SUTH + S_SUTH) / (t + S_SUTH)


def first_cell_height(rho: float, v: float, mu: float, x: float, y_plus: float = 1.0) -> float:
    """Wall-normal height [m] of the first cell for a target y+, turbulent flat plate at x."""
    re_x = rho * v * x / mu
    cf = 0.026 * re_x ** (-1.0 / 7.0)
    u_tau = math.sqrt(0.5 * cf * v * v)
    return y_plus * mu / (rho * u_tau)


def main() -> None:
    print(f"R = {R_AIR:.3f} J/(kg K), gamma = {GAMMA:.4f}, A_ref = {A_REF} m2")
    for mach, alt, p, t, f_p, f_v in CASES:
        rho = p / (R_AIR * t)
        a = math.sqrt(GAMMA * R_AIR * t)
        v = mach * a
        mu = sutherland(t)
        q = 0.5 * rho * v * v
        stag = 1.0 + 0.5 * (GAMMA - 1.0) * mach * mach
        k = 1.5 * (TURB_INTENSITY * v) ** 2
        omega = rho * k / (mu * TURB_VISC_RATIO)
        print(f"\nM = {mach} ({alt} m): p = {p:.0f} Pa, T = {t} K")
        print(f"  rho = {rho:.5f} kg/m3, a = {a:.2f} m/s, V = {v:.2f} m/s, mu = {mu:.4e} Pa s")
        print(f"  q = {q:.0f} Pa, Re_D = {rho * v * D_REF / mu:.3e}, "
              f"Re_shoulder = {rho * v * X_SHOULDER / mu:.3e}")
        print(f"  T0 = {t * stag:.1f} K, p0 = {p * stag ** (GAMMA / (GAMMA - 1.0)):.0f} Pa")
        print(f"  turbulence: k = {k:.3f} m2/s2, omega = {omega:.4e} 1/s")
        print(f"  first cell for y+ = 1 at the shoulder: "
              f"{first_cell_height(rho, v, mu, X_SHOULDER) * 1e6:.2f} um")
        print(f"  Cd wall_nose = {(f_p + f_v) / (q * A_REF):.5f} "
              f"(pressure {f_p / (q * A_REF):.5f}, viscous {f_v / (q * A_REF):.5f})")


if __name__ == "__main__":
    main()

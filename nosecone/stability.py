"""Barrowman static stability: fin normal-force slope and the static-margin change of a nose."""
from __future__ import annotations

import numpy as np

from .config import Config
from .flight import burnout_mass
from .geometry import Profile, volume_and_area


def fin_cna_xcp(cfg: Config) -> tuple[float, float]:
    """Barrowman normal-force slope of the fin set and its CP (from the fin root leading edge) [m]."""
    f, d = cfg.fins, cfg.vehicle.diameter
    cr, ct, s, xt, n = f.root_chord, f.tip_chord, f.span, f.sweep, f.count
    lm = np.hypot(s, xt + ct / 2 - cr / 2)
    cna = (1 + (d / 2) / (s + d / 2)) * (4 * n * (s / d) ** 2) / (1 + np.sqrt(1 + (2 * lm / (cr + ct)) ** 2))
    xf = xt * (cr + 2 * ct) / (3 * (cr + ct)) + (cr + ct - cr * ct / (cr + ct)) / 6
    return cna, xf


def nose_mass(cfg: Config, profile: Profile, L: float) -> float:
    """Shell (wetted area x areal density) plus tip insert [kg]."""
    _, S = volume_and_area(profile, L, cfg.R)
    return cfg.structure.shell_areal_density * S + cfg.structure.tip_mass


def static_margin_shift(cfg: Config, profile: Profile, fineness: float, baseline: Profile,
                        baseline_fineness: float) -> float:
    """Change of the subsonic static margin [cal] relative to the baseline nose.

    Positions are measured from the aft end so that a longer nose does not shift the body and
    the fins; the CG moves with the extra nose mass (nominal burn-out mass). Nose CP from
    Barrowman, x_cp = L - V/A_b from the apex, C_N_alpha = 2.
    """
    R, D = cfg.R, cfg.vehicle.diameter
    cna_f, xf = fin_cna_xcp(cfg)
    L_body = cfg.vehicle.length - cfg.vehicle.reference_nose_length
    x_f_aft = cfg.fins.root_chord - xf                  # fins flush with the aft end

    def cp_cg(pr: Profile, ff: float) -> tuple[float, float, float]:
        L = 2 * R * ff
        V, _ = volume_and_area(pr, L, R)
        x_n_aft = L_body + V / (np.pi * R**2)           # nose CP: L - V/A from the tip
        x_cp = (2 * x_n_aft + cna_f * x_f_aft) / (2 + cna_f)
        m = nose_mass(cfg, pr, L)
        x_m = L_body + cfg.structure.shell_cg_fraction * L
        return x_cp, m, x_m

    cp0, m0, xm0 = cp_cg(baseline, baseline_fineness)
    cp1, m1, xm1 = cp_cg(profile, fineness)
    m_bo = burnout_mass(cfg, "isp")
    dx_cg = (m1 * xm1 - m0 * xm0) / (m_bo + m1 - m0)
    # static margin = (x_cg - x_cp) / D measured from the aft end (CG ahead of CP)
    return ((dx_cg) - (cp1 - cp0)) / D

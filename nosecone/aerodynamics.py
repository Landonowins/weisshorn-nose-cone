"""Nose wave-drag model C_D,w(M) for any profile and fineness, built from the CFD database.

Supersonic: Euler values interpolated linearly in Mach and in fineness, on C_D,w (L/D)^2, which
is nearly constant for slender bodies (slender-body theory gives C_D,w ~ 1/(L/D)^2); with a
single fineness in the database the product is held constant. Transonic: smooth (cubic) rise
from zero at ``transonic_onset`` to the value at ``transonic_match`` reached at
``transonic_full``, then flat. The transonic part is a modelling assumption (RANS needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class TransonicModel:
    """Parameters of the transonic rise (Mach numbers)."""

    onset: float = 0.8
    full: float = 1.0
    match: float = 1.1


def select(df: pd.DataFrame, studies: Sequence[str], min_fineness: float = 0.0, max_fineness: float = 5.0,
           rn: Optional[float] = 0.0) -> pd.DataFrame:
    """Database rows of the given studies and fineness range (and tip radius, if given)."""
    d = df[df.study.isin(list(studies)) & (df.fineness >= min_fineness - 1e-9) & (df.fineness <= max_fineness + 1e-9)]
    if rn is not None:
        d = d[np.isclose(d.rn, rn)]
    return d


def select_from_config(df: pd.DataFrame, cfg: Config, rn: Optional[float] = 0.0) -> pd.DataFrame:
    """:func:`select` with the studies and fineness range of the configuration."""
    a = cfg.aerodynamics
    return select(df, a.studies, a.min_fineness, a.max_fineness, rn)


class WaveDrag:
    """C_D,w(M) of one profile at an arbitrary fineness (callable on scalars or arrays).

    Args:
        df: database rows (see :func:`select`).
        profile: profile name (the database column 'shape').
        fineness: L/D at which the model is wanted.
        machs: supersonic Mach numbers of the table.
        transonic: transonic-rise parameters.
        rn: tip radius r_n / R of the rows to use.
        grid: grid level of the rows to use.
    """

    def __init__(self, df: pd.DataFrame, profile: str, fineness: float, machs: Sequence[float],
                 transonic: TransonicModel = TransonicModel(), rn: float = 0.0, grid: str = "medium"):
        d = df[(df["shape"] == profile) & np.isclose(df.rn, rn) & (df.grid == grid)].sort_values(["fineness", "mach"])
        # only fineness values with a usable Mach curve; with a single one, C_D f^2 = const
        counts = d.groupby("fineness").mach.nunique()
        fs = np.sort(counts[counts >= 2].index.values)
        if len(fs) == 0:
            raise ValueError(f"no CFD data for profile '{profile}' (r_n/R = {rn:g}, grid {grid})")
        d = d[d.fineness.isin(fs)]
        self.profile, self.fineness, self.transonic = profile, fineness, transonic
        self.n_fineness = len(fs)
        self.fineness_data = fs
        self.machs = np.asarray(machs, dtype=float)
        # each fineness on the common Mach grid (some geometries also have M = 1.3)
        tab = np.array([np.interp(self.machs, d[d.fineness == ff].mach, d[d.fineness == ff].cd) * ff**2
                        for ff in fs])
        w = np.array([np.interp(fineness, fs, tab[:, k]) for k in range(len(self.machs))])
        self.cd_sup = w / fineness**2

    def __call__(self, M):
        t = self.transonic
        M = np.asarray(M, dtype=float)
        c11 = np.interp(t.match, self.machs, self.cd_sup)
        s = np.clip((M - t.onset) / (t.full - t.onset), 0, 1)
        trans = c11 * s * s * (3 - 2 * s)
        sup = np.interp(M, self.machs, self.cd_sup)
        return np.where(M >= t.match, sup, trans)


def transonic_from_config(cfg: Config, onset: Optional[float] = None) -> TransonicModel:
    """Transonic model of the configuration (optionally with another onset Mach number)."""
    a = cfg.aerodynamics
    return TransonicModel(a.transonic_onset if onset is None else onset, a.transonic_full, a.transonic_match)


def wave_drag(cfg: Config, df: pd.DataFrame, profile: str, fineness: float, rn: float = 0.0,
              onset: Optional[float] = None) -> WaveDrag:
    """Wave-drag model of a profile with the settings of the configuration."""
    return WaveDrag(df, profile, fineness, cfg.aerodynamics.machs, transonic_from_config(cfg, onset), rn,
                    cfg.aerodynamics.grid)


def scaled(cdw: Callable, factor_below: float, match: float) -> Callable:
    """Wave drag multiplied by `factor_below` below the Mach number `match` (robustness study)."""
    return lambda M, b=cdw: b(M) * np.where(np.asarray(M) < match, factor_below, 1.0)


def missing_cases(df: pd.DataFrame, cfg: Config, profiles: Sequence[str], fineness: Sequence[float],
                  rns: Sequence[float]) -> list[tuple[str, float, float]]:
    """(profile, fineness, rn) combinations for which no wave-drag model can be built."""
    out = []
    for p in profiles:
        for rn in rns:
            try:
                WaveDrag(select_from_config(df, cfg, rn=None), p, 4.0, cfg.aerodynamics.machs, rn=rn,
                         grid=cfg.aerodynamics.grid)
            except ValueError:
                out += [(p, f, rn) for f in fineness]
    return out

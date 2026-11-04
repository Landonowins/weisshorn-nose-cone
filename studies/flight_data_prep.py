"""Download and normalise the measured flights used to validate the flight model.

The flight logs, thrust curves and rocket data come from the RocketPy repository (MIT licence),
which publishes them with the permission of each team (see data/flights/README.md). This script
downloads the raw files of one pinned commit into output/flight_data_raw/ (not in git) and writes,
for each rocket, into data/flights/<rocket>/:

* flight.csv: t_s, z_m (above the pad) and, where the log has it, V_m_s, from lift-off to a few
  seconds after apogee, resampled at 20 Hz;
* thrust.csv: t_s, F_N of the motor, from the release of the rocket (t = 0).

The rocket geometry, masses and launch conditions are in configs/validation/<rocket>.yaml.

Usage:
    python studies/flight_data_prep.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "flight_data_raw"
OUT = ROOT / "data" / "flights"
COMMIT = "9bd6ad3af8f97bafa3201d4e70eba2877e66e040"       # RocketPy master, 22.07.2026
BASE = f"https://raw.githubusercontent.com/RocketPy-Team/RocketPy/{COMMIT}/data/"
DT = 0.05                                                   # output sampling [s]
AFTER_APOGEE = 5.0                                          # kept after apogee [s]

# rocket -> (flight log, thrust curve, hold-down time [s] before release)
SOURCES = {
    "bella_lui": ("rockets/EPFL_Bella_Lui/bella_lui_flight_data_filtered.csv", "motors/aerotech/AeroTech_K828FJ.eng", 0.0),
    "cavour": ("rockets/polito/altimeter_cavour.csv", "motors/cesaroni/Cesaroni_3618L995-P.eng", 0.0),
    "halcyon": ("rockets/astg/altimeter_halcyon.csv", "rockets/astg/engine_Halcyon_4thHotfire.eng", 0.0),
    "andromeda": ("rockets/andromeda/flight_data.csv", "rockets/andromeda/thrust_curve.csv", 0.0),
    "hedy": ("rockets/hedy/cats_tust/flightInfo.csv", "rockets/hedy/hedy_thrust_euroc_launch.csv", 1.4),
    "defiance": (None, "rockets/defiance/Thrust_curve.csv", 0.0),       # apogee only
}


def fetch(rel: str) -> Path:
    """Raw file from the pinned RocketPy commit (downloaded once)."""
    dst = RAW / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + rel, dst)
    return dst


def read_log(name: str, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Time [s], height above the pad [m] and, if logged, speed [m/s]."""
    d = pd.read_csv(path)
    if name == "bella_lui":
        return d["time aprox (s)"].values, d["z (m)"].values, d["v (m/s)"].values
    if name == "cavour":
        return d.ts.values, d["altitude[m]"].values, d["velocity[m/s]"].values
    if name == "halcyon":
        return d.ts.values, d.filtered_altitude_AGL.values, None
    if name == "andromeda":
        return d["t(s)"].values, d["alt(m)"].values, d["vz(m/s)"].values
    if name == "hedy":
        return d.ts.values, d.height.values, d.velocity.values
    raise KeyError(name)


def flight_csv(name: str, rel: str) -> pd.DataFrame:
    t, z, v = read_log(name, fetch(rel))
    o = np.argsort(t, kind="stable")
    t, z = np.asarray(t, float)[o], np.asarray(z, float)[o]
    z = z - (np.median(z[t <= 0]) if (t <= 0).any() else z[0])        # height above the pad
    v = None if v is None else np.asarray(v, float)[o]
    t_end = t[np.argmax(z)] + AFTER_APOGEE
    g = np.arange(0.0, t_end, DT)
    out = pd.DataFrame(dict(t_s=g, z_m=np.interp(g, t, z)))
    out["V_m_s"] = np.interp(g, t, v) if v is not None else np.nan
    return out.round(3)


def thrust_csv(rel: str, holddown: float) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(ROOT))
    from nosecone.flight import load_thrust_curve
    t, F = load_thrust_curve(fetch(rel))
    if holddown > 0:                       # the rocket is released at t = holddown
        F0 = np.interp(holddown, t, F)
        keep = t > holddown
        t, F = np.r_[holddown, t[keep]] - holddown, np.r_[F0, F[keep]]
    return pd.DataFrame(dict(t_s=t, F_N=F)).round(4)


def main() -> None:
    for name, (log, motor, holddown) in SOURCES.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        thrust_csv(motor, holddown).to_csv(d / "thrust.csv", index=False)
        if name == "hedy":
            # second reading of the same curve, as in the RocketPy example: from ignition, cut at the
            # burn time of the tanks (8.05 s); it brackets the thrust with the hold-down reading above
            tc = thrust_csv(motor, 0.0)
            t_cut = 8.05
            F_cut = np.interp(t_cut, tc.t_s, tc.F_N)
            tc = pd.concat([tc[tc.t_s < t_cut], pd.DataFrame(dict(t_s=[t_cut, t_cut + 1e-3], F_N=[F_cut, 0.0]))])
            tc.round(4).to_csv(d / "thrust_from_ignition.csv", index=False)
        if log:
            f = flight_csv(name, log)
            f.to_csv(d / "flight.csv", index=False)
            print(f"{name:10s} apogee {f.z_m.max():7.1f} m at {f.t_s[f.z_m.idxmax()]:5.1f} s, "
                  f"{len(f)} samples{'' if f.V_m_s.notna().any() else ' (no speed in the log)'}")
        else:
            print(f"{name:10s} thrust curve only")


if __name__ == "__main__":
    main()

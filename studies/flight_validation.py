"""Validation of the flight model against measured flights of other student rockets.

Each rocket of configs/validation/ is flown with the same drag build-up, atmosphere and numerics as
the Weisshorn study, but with the build-up as is (k = 1) and its own measured thrust curve and
masses. The result is compared with the measured flight (data/flights/, from the RocketPy
repository; run studies/flight_data_prep.py first):

* the apogee with k = 1, and the factor k on the non-nose drag that reproduces the measured apogee,
  to compare with the k = 0.565 the Weisshorn study needs to reach its 9 km target;
* the altitude and speed against time (measured time shifted so that both pass 10 m together).

Writes output/validation/: flight_validation.csv, fig_validation_altitude, fig_validation_speed,
fig_validation_apogee (PDF and PNG), and prints the table.

Usage:
    python studies/flight_validation.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nosecone import config as cfgmod  # noqa: E402
from nosecone.design import Study  # noqa: E402
from nosecone.flight import simulate  # noqa: E402
from nosecone.pipeline import load_databases  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402

CFG = ROOT / "configs" / "validation"
DATA = ROOT / "data" / "flights"
OUT = ROOT / "output" / "validation"
WEISSHORN = ROOT / "configs" / "weisshorn_720.yaml"
Z_ALIGN = 10.0                  # the measured time is shifted so that both curves pass this height together [m]


@dataclass(frozen=True)
class Flight:
    key: str                    # config file (configs/validation/<key>.yaml)
    label: str
    event: str
    propulsion: str
    apogee: float               # official / recorded apogee above the pad [m]
    log: str | None             # folder in data/flights with flight.csv, None if apogee only


FLIGHTS = [
    Flight("bella_lui", "Bella Lui", "EPFL, Kaltbrunn 2020", "solid", 458.97, "bella_lui"),
    Flight("cavour", "Cavour", "PoliTo, EuRoC 2023", "solid", 2789.0, "cavour"),
    Flight("halcyon", "Halcyon", "TU Graz, EuRoC 2023", "hybrid", 3450.0, "halcyon"),
    Flight("andromeda", "Andromeda", "AUTh, EuRoC 2022", "solid", 3443.0, "andromeda"),
    Flight("hedy", "Hedy (from release)", "TU Wien, EuRoC 2025", "LOX/ethanol", 5231.53, "hedy"),
    Flight("hedy_ignition", "Hedy (from ignition)", "TU Wien, EuRoC 2025", "LOX/ethanol", 5231.53, "hedy"),
    Flight("defiance", "Defiance", "Toronto, Launch Canada 2024", "hybrid", 9308.32, None),
]


def run(f: Flight, db: pd.DataFrame) -> dict:
    """Model flight with k = 1, the k that matches the measured apogee, and the comparison."""
    cfg = cfgmod.load(CFG / f"{f.key}.yaml")
    st = Study(cfg, db, k=1.0)
    prof, fin = st.design[:2]
    fly = lambda k, dt=None: simulate(cfg, st.vehicle(prof, fin, k=k), dt_out=dt)  # noqa: E731
    tr = fly(1.0, 0.02)
    k_fit = brentq(lambda k: fly(k)["apogee"] - f.apogee, 0.2, 3.0, xtol=1e-4)
    res = dict(flight=f.label, event=f.event, propulsion=f.propulsion, apogee_measured=f.apogee,
               apogee_model=tr["apogee"], error_pct=100 * (tr["apogee"] / f.apogee - 1), k_fit=k_fit,
               M_max_model=tr["M"].max(), t_apogee_model=tr["t_apogee"], tr=tr)
    if f.log:
        m = pd.read_csv(DATA / f.log / "flight.csv")
        shift = np.interp(Z_ALIGN, tr["z"][: np.argmax(tr["z"])], tr["t"][: np.argmax(tr["z"])]) \
            - np.interp(Z_ALIGN, m.z_m[: m.z_m.idxmax()], m.t_s[: m.z_m.idxmax()])
        m = m.assign(t=m.t_s + shift)
        res.update(meas=m, t_apogee_measured=m.t[m.z_m.idxmax()])
        if m.V_m_s.notna().any():
            up = m[m.t <= res["t_apogee_measured"]]
            res["V_max_measured"] = up.V_m_s.max()
            res["V_max_model"] = tr["V"].max()
        # altitude difference over the climb, relative to the apogee
        g = m[(m.t > 0) & (m.t <= res["t_apogee_measured"])]
        res["z_rms_pct"] = 100 * np.sqrt(np.mean((np.interp(g.t, tr["t"], tr["z"]) - g.z_m) ** 2)) / f.apogee
    return res


def fig_altitude(results: list[dict], out: Path) -> None:
    logs = [r for r in results if "meas" in r and "ignition" not in r["flight"]]
    hedy_ign = next((r for r in results if "ignition" in r["flight"]), None)
    fig, axs = plt.subplots(2, 3, figsize=(fs.WIDTH, 0.62 * fs.WIDTH))
    for a, r in zip(axs.flat, logs):
        a.plot(r["meas"].t, r["meas"].z_m / 1e3, color=fs.INK, lw=1.2, label="measured")
        a.plot(r["tr"]["t"], r["tr"]["z"] / 1e3, color="#d03b3b", lw=1.0, ls=(0, (4, 2)), label="model, $k = 1$")
        if r["flight"].startswith("Hedy") and hedy_ign is not None:
            a.plot(hedy_ign["tr"]["t"], hedy_ign["tr"]["z"] / 1e3, color="#2a78d6", lw=1.0, ls=(0, (1, 1.5)),
                   label="model, curve from ignition")
            a.legend(fontsize=6, loc="lower right")
        a.set_title(f"{r['flight'].split(' (')[0]} ({r['event'].split(', ')[1]})", fontsize=8)
        a.set_xlabel(r"$t$ [s]")
    for a in axs[:, 0]:
        a.set_ylabel(r"height [km]")
    axs.flat[0].legend(fontsize=6, loc="lower right")
    d = next(r for r in results if r["flight"] == "Defiance")
    a = axs.flat[len(logs)]
    a.plot(d["tr"]["t"], d["tr"]["z"] / 1e3, color="#d03b3b", lw=1.0, ls=(0, (4, 2)))
    a.axhline(d["apogee_measured"] / 1e3, color=fs.INK, lw=1.2)
    a.text(1, d["apogee_measured"] / 1e3 * 0.93, "measured apogee", fontsize=6)
    a.set_title("Defiance (Launch Canada 2024)", fontsize=8)
    a.set_xlabel(r"$t$ [s]")
    for a in axs.flat[len(logs) + 1:]:
        a.axis("off")
    fig.tight_layout()
    fs.save(fig, "fig_validation_altitude", out)


def fig_speed(results: list[dict], out: Path) -> None:
    logs = [r for r in results if "V_max_measured" in r and "ignition" not in r["flight"]]
    fig, axs = plt.subplots(1, len(logs), figsize=(fs.WIDTH, 0.32 * fs.WIDTH))
    for a, r in zip(np.atleast_1d(axs), logs):
        m = r["meas"][r["meas"].t <= r["t_apogee_measured"]]
        a.plot(m.t, m.V_m_s, color=fs.INK, lw=1.2, label="measured")
        a.plot(r["tr"]["t"], r["tr"]["V"], color="#d03b3b", lw=1.0, ls=(0, (4, 2)), label="model, $k = 1$")
        a.set_title(r["flight"].split(" (")[0], fontsize=8)
        a.set_xlabel(r"$t$ [s]")
        a.set_xlim(0, r["t_apogee_measured"])
    np.atleast_1d(axs)[0].set_ylabel(r"speed [m/s]")
    np.atleast_1d(axs)[0].legend(fontsize=6)
    fig.tight_layout()
    fs.save(fig, "fig_validation_speed", out)


def fig_apogee(results: list[dict], k_weisshorn: float, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.42 * fs.WIDTH))
    zmax = 1.1 * max(max(r["apogee_measured"], r["apogee_model"]) for r in results) / 1e3
    a.plot([0, zmax], [0, zmax], color=fs.GRID, lw=0.8)
    a.fill_between([0, zmax], [0, 0.9 * zmax], [0, 1.1 * zmax], color=fs.GRID, alpha=0.35, lw=0)
    offset = {"Andromeda": (-4, 1, "right"), "Halcyon": (5, -7, "left")}   # labels of the two close points
    for r in results:
        a.plot(r["apogee_measured"] / 1e3, r["apogee_model"] / 1e3, "o", ms=4,
               color="#2a78d6" if "ignition" in r["flight"] else "#d03b3b")
        dx, dy, ha = offset.get(r["flight"], (4, -3, "left"))
        a.annotate(r["flight"].replace(" (from release)", "").replace(" (from ignition)", " (ign.)"),
                   (r["apogee_measured"] / 1e3, r["apogee_model"] / 1e3), fontsize=6,
                   xytext=(dx, dy), textcoords="offset points", ha=ha)
    a.set(xlim=(0, zmax), ylim=(0, zmax), xlabel="measured apogee [km]", ylabel=r"model apogee, $k = 1$ [km]",
          title=r"(a) apogee (grey band: $\pm 10\,\%$)")
    names = [r["flight"].replace(" (from release)", "\n(release)").replace(" (from ignition)", "\n(ignition)")
             for r in results]
    y = np.arange(len(results))
    b.barh(y, [r["k_fit"] for r in results], color="#9aa7b4", height=0.6)
    b.axvline(1.0, color=fs.INK2, lw=0.8)
    b.axvline(k_weisshorn, color="#d03b3b", lw=1.2, ls=(0, (4, 2)),
              label=f"Weisshorn, $k = {k_weisshorn:.3f}$\n(9 km target)")
    b.legend(fontsize=6, loc="upper right", framealpha=0.95, handlelength=1.5)
    b.set_yticks(y, names, fontsize=6)
    b.invert_yaxis()
    b.set(xlim=(0, 1.6), xlabel=r"$k$ that matches the measured apogee", title="(b) factor on the non-nose drag")
    fig.tight_layout()
    fs.save(fig, "fig_validation_apogee", out)


def main() -> None:
    fs.setup()
    w = cfgmod.load(WEISSHORN)
    db = load_databases(w)
    k_w = Study(w, db).k
    results = [run(f, db) for f in FLIGHTS]
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["flight", "event", "propulsion", "apogee_measured", "apogee_model", "error_pct", "k_fit", "M_max_model",
            "t_apogee_measured", "t_apogee_model", "V_max_measured", "V_max_model", "z_rms_pct"]
    table = pd.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in results])
    table.round(4).to_csv(OUT / "flight_validation.csv", index=False)
    fig_altitude(results, OUT)
    fig_speed(results, OUT)
    fig_apogee(results, k_w, OUT)
    pd.set_option("display.width", 200)
    print(table.drop(columns=["event"]).round(3).to_string(index=False))
    print(f"\nWeisshorn (weisshorn_720.yaml): k = {k_w:.4f} to reach the 9 km target apogee")


if __name__ == "__main__":
    main()

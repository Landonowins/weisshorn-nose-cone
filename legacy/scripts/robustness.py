"""Robustness of the shape ranking to the uncertain modelling choices.

For every shape at L/D = 4.5 the apogee is recomputed under perturbed assumptions and
reported relative to the Von Karman nose in the same scenario. Writes results/robustness.csv.
"""
import os
import numpy as np
import pandas as pd

import analysis as an
from flight import simulate
from geometry import SHAPES

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
F = 4.5

SCENARIOS = {
    "nominal": dict(),
    "drag rise from M = 0.75": dict(onset=0.75),
    "drag rise from M = 0.85": dict(onset=0.85),
    "transonic peak +20 %": dict(trans=1.2),
    "transonic peak -20 %": dict(trans=0.8),
    "rest of vehicle drag +20 %": dict(kfac=1.2),
    "LVO masses, 30 C day": dict(mass="lvo", Tg=303.15),
}


def apogee(df, shape, k, onset=0.8, trans=1.0, kfac=1.0, mass="isp", Tg=288.15):
    an.M_ONSET = onset
    veh = an.make_vehicle(df, shape, F, k * kfac)
    base = veh.cdw
    veh.cdw = lambda M, b=base: b(M) * np.where(np.asarray(M) < 1.1, trans, 1.0)
    dm = an.nose_mass(shape, 2 * an.R * F) - an.nose_mass(*an.BASELINE[:1], 2 * an.R * an.BASELINE[1])
    z = simulate(veh, m_nose_delta=dm, mass_model=mass, T_ground=Tg)["apogee"]
    an.M_ONSET = 0.8
    return z


def main():
    df = an.load_sweep()
    k = float(open(os.path.join(RES, "calibration.txt")).readline().split("=")[1])
    rows = []
    for name, kw in SCENARIOS.items():
        z = {s: apogee(df, s, k, **kw) for s in SHAPES}
        for s in SHAPES:
            rows.append(dict(scenario=name, shape=s, apogee=z[s], dz_vs_vk=z[s] - z["vonkarman"]))
        print(name, {s: round(z[s] - z["vonkarman"]) for s in SHAPES}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "robustness.csv"), index=False)


if __name__ == "__main__":
    main()

"""Thermal profile of the 720 mm noses along the whole shell, for 2027_C_FD_REPORT-5.

Runs the calibrated shell model of the report (1D conduction through the CFRP, heated outside, adiabatic
inside; h and T_r calibrated on Fluent) at stations from 10 mm to the shoulder, for both designs and
both flight cases, and writes

* figures/fig_thermal_profile: peak outer- and inner-face temperature along the nose;
* figures/fig_thermal_map: outer- and inner-face temperature against time and x (design A);
* data/{A,B}_thermal_x.csv: peak temperatures and their times at every station;
* sections/numbers_thermal.tex: the values quoted in the text, as \\val{th-...}.

Usage (after report_720/figures.py, whose inputs it reuses):
    python report_720/thermal_profile.py configs/weisshorn_720.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

import figures as F                     # also sets up the paths and matplotlib backend
import total_length as tl
from nosecone import heating as ht
from nosecone.plotting import style as fs

HERE = Path(__file__).resolve().parent
X_MM = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 350, 400, 450, 500,
                 550, 600, 650])     # the shoulder of each design is added
CASE_LAB = {"nominal": "nominal", "hot": "hot day"}


def profiles(d: F.Design, mdl: tl.Model) -> dict:
    """Outer and inner face temperature [K] at every station: {case: (x_m, t, T_out[nx, nt], T_in[nx, nt])}."""
    h = mdl.cfg.heating
    mat = mdl.st.materials()[h.shell_material]
    x = np.append(1e-3 * X_MM, d.xS)
    res = {}
    for c in F.CASES:
        tr = d.tr[c]
        To, Ti = [], []
        for xi in x:
            wr = ht.wall_response(tr, mat, h.shell_thickness, float(xi), n=h.wall_nodes, **ht.calibration(h))
            To.append(np.concatenate([[tr["T"][0]], wr["T_out"]]))
            Ti.append(np.concatenate([[tr["T"][0]], wr["T_in"]]))
            print(f"{d.study} {c:7s} x = {1e3 * xi:5.0f} mm: T_out {max(To[-1]) - 273.15:5.1f} C, "
                  f"T_in {max(Ti[-1]) - 273.15:5.1f} C", flush=True)
        res[c] = (x, np.asarray(tr["t"]), np.array(To), np.array(Ti))
    return res


def fig_profile(prof: dict, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.42 * fs.WIDTH), sharey=True)
    for a, (k, title, lim, lab) in zip(axs, [(2, "(a) outer face", F.LIM_DP490, "DP490 120 $^\\circ$C"),
                                             (3, "(b) inner face", F.LIM_REQ10, "Req\\_10 67 $^\\circ$C")]):
        a.axvspan(0, 80, color="0.92", lw=0)
        a.text(40, 4, "Al tip insert", ha="center", va="bottom", rotation=90, fontsize=7, color=fs.INK2)
        for s, p in prof.items():
            for c, (x, t, *T) in p.items():
                a.plot(1e3 * x, T[k - 2].max(axis=1) - 273.15, color=F.SHAPE_COL[s], ls=F.CASE_LS[c], lw=1.1)
        a.axhline(lim, color=fs.INK, lw=0.6, ls=(0, (4, 2)))
        a.text(715, lim + 2 if lim > 100 else lim - 7, lab, ha="right", fontsize=7)
        a.set(xlim=(0, 720), xlabel=r"$x$ from the tip [mm]", title=title)
    axs[0].set(ylim=(0, 130), ylabel=r"highest temperature in flight [$^\circ$C]")
    fig.legend(handles=[Line2D([], [], color=F.SHAPE_COL[s], label=f"design {s}") for s in prof]
               + [Line2D([], [], color=fs.INK2, ls=F.CASE_LS[c], label=CASE_LAB[c]) for c in F.CASES],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7)
    fs.save(fig, "fig_thermal_profile", out)


def fig_map(p: dict, out: Path) -> None:
    """Temperature against time and x for one design: rows nominal / hot day, columns outer / inner face."""
    fig, axs = plt.subplots(2, 2, figsize=(fs.WIDTH, 0.72 * fs.WIDTH), sharex=True, sharey=True)
    levels = np.arange(10, 125, 5)
    for i, c in enumerate(F.CASES):
        x, t, To, Ti = p[c]
        for j, (T, lim, name) in enumerate([(To, F.LIM_DP490, "outer face"), (Ti, F.LIM_REQ10, "inner face")]):
            a = axs[i, j]
            cf = a.contourf(t, 1e3 * x, T - 273.15, levels=levels, cmap="inferno", extend="both")
            cf.set_rasterized(True)             # no white seams between the filled levels in the PDF
            cs = a.contour(t, 1e3 * x, T - 273.15, levels=[lim], colors="w", linewidths=0.9, linestyles="--")
            a.clabel(cs, fmt=lambda v: f"{v:.0f} °C", fontsize=6)
            a.set(xlim=(0, 30), ylim=(10, 1e3 * x[-1]), yscale="log")
            a.set_yticks([10, 20, 50, 100, 200, 500])
            a.yaxis.set_major_formatter(ScalarFormatter())
            a.yaxis.set_minor_formatter(NullFormatter())
            a.set_title(f"{name}, {CASE_LAB[c]}", fontsize=8)
            if i == 1:
                a.set_xlabel(r"$t$ [s]")
            if j == 0:
                a.set_ylabel(r"$x$ from the tip [mm]")
    cb = fig.colorbar(cf, ax=axs, shrink=0.9, pad=0.02)
    cb.set_label(r"shell temperature [$^\circ$C]")
    fs.save(fig, "fig_thermal_map", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    a = ap.parse_args()
    fs.setup()
    mdl = tl.Model(Path(a.config))
    summary = json.loads((Path(mdl.cfg.project.output_dir) / "final" / "summary.json").read_text(encoding="utf-8"))
    fsm = F.free_stream()
    des = {s: F.Design(mdl, s, summary, fsm) for s in ("A", "B")}
    prof = {s: profiles(d, mdl) for s, d in des.items()}

    figs = HERE / "figures"
    fig_profile(prof, figs)
    fig_map(prof["A"], figs)

    v = {}
    for s, p in prof.items():
        rows = {}
        for c, (x, t, To, Ti) in p.items():
            rows["x_mm"] = np.round(1e3 * x, 1)
            for k, T in (("out", To), ("in", Ti)):
                rows[f"T_{k}_max_{c}_C"] = np.round(T.max(axis=1) - 273.15, 2)
                rows[f"t_T_{k}_max_{c}_s"] = np.round(t[T.argmax(axis=1)], 2)
            for st, xi in (("20", 0.02), ("mid", 0.5 * des[s].xS), ("shoulder", des[s].xS)):
                n = int(np.argmin(abs(x - xi)))
                v[f"th-{s}-{c}-Tout-{st}"] = f"{To[n].max() - 273.15:.0f}"
                v[f"th-{s}-{c}-Tin-{st}"] = f"{Ti[n].max() - 273.15:.0f}"
            v[f"th-{s}-{c}-Tin-lo"] = f"{Ti.max(axis=1).min() - 273.15:.0f}"
            v[f"th-{s}-{c}-Tin-hi"] = f"{Ti.max(axis=1).max() - 273.15:.0f}"
            v[f"th-{s}-{c}-Tout-hi"] = f"{To.max(axis=1).max() - 273.15:.0f}"
            above = t[(Ti > F.LIM_REQ10 + 273.15).any(axis=0)]
            v[f"th-{s}-{c}-req10-t0"] = f"{above.min():.0f}" if len(above) else "--"
            v[f"th-{s}-{c}-req10-t1"] = f"{above.max():.0f}" if len(above) else "--"
        pd.DataFrame(rows).to_csv(HERE / "data" / f"{s}_thermal_x.csv", index=False)
    # largest A-B difference of the peak temperatures at the common stations (x <= 650 mm)
    n = len(X_MM)
    dAB = max(abs(prof["A"][c][k][:n].max(axis=1) - prof["B"][c][k][:n].max(axis=1)).max()
              for c in F.CASES for k in (2, 3))
    v["th-AB-diff"] = f"{dAB:.1f}"
    lines = ["% generated by report_720/thermal_profile.py, do not edit"]
    lines += [rf"\expandafter\def\csname v@{k}\endcsname{{{x}}}" for k, x in sorted(v.items())]
    (HERE / "sections" / "numbers_thermal.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k in sorted(v):
        print(f"{k:28s} {v[k]}")


if __name__ == "__main__":
    main()

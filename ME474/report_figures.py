"""The report figure that Fluent cannot draw: C_D(M) of A and B across the production runs.

Usage: python report_figures.py      (a Python with matplotlib, e.g. the Ansys CPython 3.10; reads runs/summary.csv)

Writes figures/fig_cd_mach.pdf and .png, 16 cm wide, in the style of the team's report figures
(nosecone/plotting/style.py: LV-Haack and Von Karman colours, line styles and markers):
  (a) C_D of nose + coupler (x = 0 to 720 mm for both designs, the headline decided 16 Nov), with the band where
      B is above A and the drag-rise Mach numbers
  (b) dC_D/dM at the interval midpoints, with the drag-rise threshold dC_D/dM = 0.1 (PLAN.md decision 3)
  (c) pressure part (all on the nose), (d) viscous part (nose + coupler)
The points, the drag rise and the crossovers are those of drag_curve.py. Every other figure (contours, Cp and Cf along
the wall, residuals, C_D histories) is a native Fluent plot: fluent_figs.py.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
sys.path.insert(0, str(HERE.parent))
import drag_curve as dc  # noqa: E402
from nosecone.plotting import style as fs  # noqa: E402

SHAPES = {"A": ("lvhaack", r"A, LV-Haack"), "B": ("vonkarman", r"B, Von K\'arm\'an")}
# plotted field -> drag_curve.FIELDS: the headline nose + coupler, its pressure part and its viscous part
FIELDS = {"cd": "cd_720", "cd_p": "cd_p", "cd_v": "cd_v720"}


def table():
    rows = {r["run"]: r for r in csv.DictReader(open(dc.RUNS / "summary.csv"))}
    out, missing = {s: [] for s in SHAPES}, []
    for point in dc.POINTS:
        ra, rb = rows.get(dc.run_name("A", point)), rows.get(dc.run_name("B", point))
        if not (ra and rb):
            missing.append(point)
            continue
        for s, r in (("A", ra), ("B", rb)):
            out[s].append({"mach": float(r["mach"]), **{k: dc.FIELDS[f](r) for k, f in FIELDS.items()}})
    if missing:
        print("left out (a run missing):", " ".join(missing))
    return out


def crossovers(tab):
    ms = [r["mach"] for r in tab["A"]]
    d = [a["cd"] - b["cd"] for a, b in zip(tab["A"], tab["B"])]
    return [m0 + d0 / (d0 - d1) * (m1 - m0) for m0, m1, d0, d1 in zip(ms, ms[1:], d, d[1:]) if d0 * d1 < 0]


def style(shape):
    color, ls, marker = fs.style_of(SHAPES[shape][0], [])
    return dict(color=color, ls=ls, marker=marker, ms=5, lw=1.0, mec=color, mfc="white" if shape == "B" else color)


def main():
    fs.setup()
    tab = table()
    cross = crossovers(tab)
    fig, axs = plt.subplots(2, 2, figsize=(fs.WIDTH, 0.70 * fs.WIDTH), sharex=True)
    (a, b), (c, d) = axs
    rise = {}
    for s in SHAPES:
        ms = [r["mach"] for r in tab[s]]
        cd = [r["cd"] for r in tab[s]]
        mdr, mids, slopes = dc.drag_rise(ms, cd)
        rise[s] = mdr
        a.plot(ms, cd, **style(s))
        b.plot(mids, slopes, **style(s))
        c.plot(ms, [r["cd_p"] for r in tab[s]], **style(s))
        d.plot(ms, [r["cd_v"] for r in tab[s]], **style(s))
        if mdr:
            i = next(k for k in range(len(ms) - 1) if ms[k] <= mdr <= ms[k + 1])
            cd_dr = cd[i] + (mdr - ms[i]) / (ms[i + 1] - ms[i]) * (cd[i + 1] - cd[i])
            a.plot([mdr], [cd_dr], ls="none", marker="o", ms=9, mfc="none", mec=fs.INK, mew=0.6, zorder=5)
            b.plot([mdr], [dc.SLOPE], ls="none", marker="o", ms=9, mfc="none", mec=fs.INK, mew=0.6, zorder=5)
    if len(cross) >= 2:
        for ax in (a, c):
            ax.axvspan(cross[0], cross[1], color=fs.GRID, alpha=0.55, lw=0, zorder=0)
        a.text(0.5 * (cross[0] + cross[1]), 0.93, "B above A", transform=a.get_xaxis_transform(), ha="center",
               va="top", rotation=90, fontsize=7, color=fs.INK2)
    for ax in (a, b, c, d):
        ax.axvline(1.0, color=fs.INK2, lw=0.5, ls=(0, (1, 2)), zorder=0)
    a.set_ylabel(r"$C_D$, nose + coupler")
    a.set_ylim(0, None)
    b.axhline(dc.SLOPE, color=fs.INK2, lw=0.6, ls=(0, (4, 2)), zorder=0)
    b.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
    b.set_ylabel(r"$\mathrm{d}C_D/\mathrm{d}M$ (interval midpoints)")
    txt = r"drag rise, $\mathrm{d}C_D/\mathrm{d}M = 0.1$:" + "".join(
        f"\n{s}: $M = {rise[s]:.3f}$" for s in SHAPES if rise[s])
    b.text(0.97, 0.95, txt, transform=b.transAxes, ha="right", va="top", fontsize=7, color=fs.INK)
    c.set_ylabel(r"pressure part $C_{D,p}$")
    c.set_ylim(0, None)
    d.set_ylabel(r"viscous part $C_{D,v}$")
    for ax, letter, y in zip((a, b, c, d), "abcd", (0.95, 0.95, 0.95, 0.12)):  # (d): its curves start top left
        ax.set_xlim(0.86, 1.50)
        fs.panel(ax, letter, x=0.03, y=y)
    for ax in (c, d):
        ax.set_xlabel(r"$M_\infty$")
    handles = [Line2D([], [], **style(s), label=SHAPES[s][1]) for s in SHAPES]
    handles.append(Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mec=fs.INK, mew=0.6, label="drag rise"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=1.2)
    for p in fs.save(fig, "fig_cd_mach", FIG, dpi=300):
        print("saved", p)
    print("drag rise:", ", ".join(f"{s} M {rise[s]:.3f}" for s in SHAPES if rise[s]),
          "| A and B cross at M " + ", ".join(f"{m:.3f}" for m in cross))


if __name__ == "__main__":
    main()

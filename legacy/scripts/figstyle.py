"""Common figure style: LaTeX text, 16 cm print width, 9 pt, boxed axes, inward ticks."""
import matplotlib
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDTH = 16 * CM

# validated categorical order (dataviz reference palette, light), one slot per shape,
# always paired with a line style / marker so identity survives B/W print and CVD
COLORS = {
    "cone": "#2a78d6", "ogive": "#eb6834", "parabolic": "#1baf7a", "power050": "#eda100",
    "power075": "#e87ba4", "lvhaack": "#008300", "vonkarman": "#4a3aa7",
}
STYLES = {
    "cone": (0, (1, 1.5)), "ogive": (0, (5, 2)), "parabolic": (0, (3, 1.5, 1, 1.5)),
    "power050": (0, (6, 1.5, 1, 1.5, 1, 1.5)), "power075": (0, (2, 2)),
    "lvhaack": (0, (8, 2, 2, 2)), "vonkarman": "-",
}
MARKERS = {"cone": "v", "ogive": "s", "parabolic": "D", "power050": "o", "power075": "^",
           "lvhaack": "P", "vonkarman": "*"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def setup():
    matplotlib.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}",
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 2.6, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150, "axes.grid": False,
    })


def panel(ax, letter, x=0.02, y=0.96, **kw):
    ax.text(x, y, rf"\textbf{{({letter})}}", transform=ax.transAxes, ha="left", va="top",
            color=INK, **kw)


def save(fig, name, outdir):
    import os
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200)
    plt.close(fig)

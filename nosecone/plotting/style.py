"""Figure style: 16 cm print width, 9 pt text, boxed axes, inward ticks, LaTeX text if available."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDTH = 16 * CM
"""Figure width [in] (16 cm, the text width of an A4 report)."""

# validated categorical order (dataviz reference palette, light), one slot per built-in profile,
# always paired with a line style / marker so that identity survives B/W print and colour blindness
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
EXTRA_COLORS = ["#7f3f98", "#b8860b", "#2f4f4f", "#c71585", "#556b2f", "#8b4513", "#1e90ff", "#696969"]
EXTRA_STYLES = ["-", (0, (4, 1.5)), (0, (1, 1)), (0, (6, 2, 1, 2)), (0, (3, 3)), "-", (0, (5, 1)), (0, (2, 1, 2, 3))]
EXTRA_MARKERS = ["X", "h", "<", ">", "p", "d", "8", "H"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

USETEX = True


def latex_available() -> bool:
    """True if a LaTeX installation usable by matplotlib is on the PATH."""
    return shutil.which("latex") is not None


def setup(usetex: Optional[bool] = None) -> bool:
    """Apply the style; LaTeX text if `usetex` (None: use LaTeX if it is installed). Returns the choice."""
    global USETEX
    USETEX = latex_available() if usetex is None else bool(usetex)
    matplotlib.rcParams.update({
        "text.usetex": USETEX,
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
    if not USETEX:
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        _patch_text()
    return USETEX


def _strip_command(s: str, cmd: str) -> str:
    """Remove `cmd{...}` wrappers (keeping the content), with balanced braces."""
    while True:
        i = s.find(cmd + "{")
        if i < 0:
            return s
        j, depth = i + len(cmd) + 1, 1
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        s = s[:i] + s[i + len(cmd) + 1:j - 1] + s[j:]


_PATCHED = False


def _patch_text() -> None:
    """Without LaTeX, pass every matplotlib text through :func:`tex` (labels are written for LaTeX)."""
    global _PATCHED
    if _PATCHED:
        return
    import matplotlib.text as mtext
    orig = mtext.Text.set_text

    def set_text(self, s):
        return orig(self, tex(s) if isinstance(s, str) and not USETEX else s)
    mtext.Text.set_text = set_text
    _PATCHED = True


_ACCENTS = {r"\'a": "á", r"\'e": "é", r"\'i": "í", r"\'o": "ó", r"\'u": "ú", r"\`a": "à", r"\`e": "è",
            r"\"a": "ä", r"\"o": "ö", r"\"u": "ü"}


def tex(s: str) -> str:
    """Make a LaTeX label printable by matplotlib's built-in mathtext (identity when LaTeX is used).

    Text segments: accents, \\textbf, \\% and spacing commands are converted; math segments
    ($...$): \\tfrac12 -> \\frac{1}{2}.
    """
    if USETEX or not isinstance(s, str):
        return s
    s = _strip_command(s, r"\textbf")
    parts = s.split("$")
    for i, seg in enumerate(parts):
        if i % 2 == 0:           # text
            for k, v in _ACCENTS.items():
                seg = seg.replace(k, v)
            seg = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", seg)
            seg = seg.replace(r"\%", "%").replace(r"\,", " ").replace(r"\ ", " ")
        else:                    # math
            seg = re.sub(r"\\[td]frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", seg)
            seg = seg.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
        parts[i] = seg
    return "$".join(parts)


def panel(ax: Any, letter: str, x: float = 0.02, y: float = 0.96, **kw: Any) -> None:
    """Panel label (a), (b), ... in axes coordinates."""
    ax.text(x, y, tex(rf"\textbf{{({letter})}}"), transform=ax.transAxes, ha="left", va="top", color=INK,
            **({"fontweight": "bold"} if not USETEX else {}), **kw)


def save(fig: Any, name: str, outdir: Path, formats: Iterable[str] = ("pdf", "png"), dpi: int = 200) -> list[Path]:
    """Save a figure in the requested formats and close it."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        if fmt == "png":
            fig.savefig(p, dpi=dpi)
        else:
            fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def style_of(name: str, order: list[str]) -> tuple[str, object, str]:
    """(colour, line style, marker) of a profile; custom profiles get the extra palette in order."""
    if name in COLORS:
        return COLORS[name], STYLES[name], MARKERS[name]
    extra = [n for n in order if n not in COLORS]
    i = extra.index(name) % len(EXTRA_COLORS) if name in extra else 0
    return EXTRA_COLORS[i], EXTRA_STYLES[i], EXTRA_MARKERS[i]

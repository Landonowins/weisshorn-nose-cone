"""Documentation helpers: the parameter tables of the user guide are generated from the config schema.

    python -m nosecone describe-config --latex > docs/user_guide/sections/parameters_tables.tex
"""
from __future__ import annotations

from .config import SECTIONS

_ESC = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}",
        "~": r"\~{}", "{": r"\{", "}": r"\}"}

SECTION_TITLES = {
    "project": "Project", "vehicle": "Vehicle", "engine": "Engine", "fins": "Fins", "launch": "Launch",
    "atmosphere": "Atmosphere", "flight": "Flight model", "nose": "Nose: design space and selected design",
    "aerodynamics": "Aerodynamics (wave-drag model)", "payload": "Payload", "structure": "Structure (mass model)",
    "heating": "Heating", "bluntness": "Tip bluntness", "robustness": "Robustness", "trade_matrix": "Decision matrix",
    "validation": "CFD validation plots", "cfd": "CFD solver and studies", "cad": "CAD export", "output": "Outputs",
}

UNITS = {"m^2": r"m\textsuperscript{2}", "kg/m^2": r"kg/m\textsuperscript{2}", "kg/m^3": r"kg/m\textsuperscript{3}",
         "degC": r"\textdegree C", "deg": r"\textdegree", "-": "--"}


def esc(s: str) -> str:
    """Escape LaTeX special characters."""
    return "".join(_ESC.get(c, c) for c in str(s))


def fmt_default(v: object) -> str:
    """Compact LaTeX rendering of a default value."""
    if isinstance(v, bool):
        return r"\texttt{" + ("true" if v else "false") + "}"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return r"\texttt{" + esc(v) + "}" if v else "(empty)"
    if isinstance(v, list):
        if not v:
            return "[\\,]"
        if all(isinstance(x, (int, float, str)) for x in v) and len(str(v)) <= 42:
            return r"\texttt{[" + esc(", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)) + "]}"
        return "see text"
    if isinstance(v, dict):
        return "see text" if v else r"\{\,\}"
    return esc(v)


def config_tables_latex(rows: list[dict]) -> str:
    """One longtable per configuration section (parameter, default, unit, valid range, meaning)."""
    out = []
    for sec in SECTIONS:
        rs = [r for r in rows if r["section"] == sec]
        if not rs:
            continue
        out.append(rf"\subsection*{{\texttt{{{esc(sec)}}}: {SECTION_TITLES.get(sec, sec)}}}")
        out.append(r"\addcontentsline{toc}{subsection}{\texttt{" + esc(sec) + "}: " + SECTION_TITLES.get(sec, sec) + "}")
        out.append(r"{\footnotesize")
        out.append(r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{2.6cm}"
                   r">{\raggedright\arraybackslash}p{1.3cm}>{\raggedright\arraybackslash}p{1.9cm}"
                   r">{\raggedright\arraybackslash}p{6.1cm}@{}}")
        out.append(r"\toprule Parameter & Default & Unit & Valid & Meaning \\ \midrule \endhead \bottomrule \endfoot")
        for r in rs:
            name = r"\texttt{" + esc(r["name"]) + "}" + (r"\,$^\ast$" if r["advanced"] else "")
            unit = UNITS.get(r["unit"], esc(r["unit"]))
            valid = esc(r["valid"]) if r["valid"] else "--"
            out.append(f"{name} & {fmt_default(r['default'])} & {unit} & {valid} & {esc(r['doc'])} \\\\")
        out.append(r"\end{longtable}\addtocounter{table}{-1}}")
        out.append("")
    return "\n".join(out)

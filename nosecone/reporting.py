"""Tables and summaries: CSV files, a Markdown summary, and optional LaTeX tables and macros."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import aerodynamics as aero
from .design import Study, mach_key, matrix_weights

# names used in the LaTeX tables of the design report (fall back to the plot label)
MATRIX_NAMES = {"cone": "Cone", "ogive": "Ogive", "parabolic": "Parab.", "power050": r"Power $\tfrac12$",
                "power075": r"Power $\tfrac34$", "lvhaack": "LV-Haack", "vonkarman": r"Von K\'arm\'an"}
CATALOGUE_NAMES = dict(MATRIX_NAMES, parabolic="Parabolic")
CRITERIA = {"performance": "Performance", "integration": "Integration, {payload} fit", "manufacture": "Manufacture",
            "stability": "Stability"}


# --------------------------------------------------------------------------- plain outputs
def write_text_results(st: Study, outdir: Path, sens: dict) -> None:
    """calibration.txt and sensitivity.txt (same format as the original scripts)."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "calibration.txt").write_text(f"k = {st.k:.4f}\n")
    (outdir / "sensitivity.txt").write_text(
        f"dz/dm = {sens['dz_dm']:.1f} m/kg\ndz/dCD = {sens['dz_dcd']:.0f} m per unit CD "
        f"({sens['dz_per_cd_step']:.1f} m per {sens['cd_step']:g})\n")


def summary_markdown(st: Study, trade: pd.DataFrame, sens: dict, robust: Optional[pd.DataFrame],
                     heat: Optional[dict], blunt: Optional[list], design: Optional[dict],
                     matrix: Optional[pd.DataFrame]) -> str:
    """Human-readable summary of the run (the one table that compares the whole sweep)."""
    cfg = st.cfg
    out = [f"# {cfg.project.name}: nose-cone study summary", ""]
    out.append(f"* Diameter {1e3 * cfg.vehicle.diameter:.0f} mm; baseline {cfg.nose.baseline_profile} "
               f"L/D = {cfg.nose.baseline_fineness:g}; design {cfg.nose.design_profile} L/D = {cfg.nose.design_fineness:g}, "
               f"r_n/R = {cfg.nose.design_tip_radius_ratio:g}.")
    how = "calibrated on the target apogee" if cfg.flight.calibrate else "fixed (flight.drag_factor)"
    out.append(f"* Drag factor k = {st.k:.4f} ({how}); baseline apogee {sens['apogee_baseline']:.0f} m.")
    out.append(f"* Sensitivities: {sens['dz_dm']:.1f} m/kg; {sens['dz_dcd'] * 0.01:.1f} m per +0.01 on C_D.")
    out += ["", "## Sweep (sorted by apogee)", ""]
    cols = ["shape", "fineness"] + (["tip_radius_ratio"] if "tip_radius_ratio" in trade else []) + \
        [c for c in trade.columns if c.startswith("cd_w_")] + ["apogee", "d_apogee", "mass_g"] + \
        [c for c in ("l3U_mm", "bay_mm") if c in trade] + ["dSM_cal", "mach_max"]
    t = trade.sort_values("apogee", ascending=False)[cols]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif c.startswith("cd_w_"):
                cells.append(f"{v:.4f}")
            elif c in ("fineness", "tip_radius_ratio"):
                cells.append(f"{v:g}")
            elif c in ("dSM_cal", "mach_max"):
                cells.append(f"{v:+.3f}" if c == "dSM_cal" else f"{v:.3f}")
            else:
                cells.append(f"{v:.0f}" if c != "d_apogee" else f"{v:+.0f}")
        out.append("| " + " | ".join(cells) + " |")
    if robust is not None and len(robust):
        out += ["", f"## Robustness (apogee relative to {cfg.robustness.reference_profile} at L/D = {cfg.robustness.fineness:g}, m)", ""]
        pv = robust.pivot(index="shape", columns="scenario", values="dz_vs_vk")
        pv = pv[list(dict.fromkeys(robust.scenario))].reindex([p for p in cfg.nose.profiles if p in pv.index])
        out.append("| profile | " + " | ".join(pv.columns) + " |")
        out.append("|" + "---|" * (len(pv.columns) + 1))
        for s, r in pv.iterrows():
            out.append(f"| {s} | " + " | ".join(f"{v:+.0f}" for v in r.values) + " |")
    if matrix is not None:
        out += ["", f"## Decision matrix (L/D = {cfg.trade_matrix.fineness:g}, scores 1-5)", ""]
        out.append("| criterion | " + " | ".join(matrix.columns) + " |")
        out.append("|" + "---|" * (len(matrix.columns) + 1))
        for c, r in matrix.iterrows():
            out.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
    if blunt:
        out += ["", "## Tip bluntness", ""]
        for b in blunt:
            out.append(f"* r_n/R = {b['rn_over_R']:g}: wave drag {b['drag_cost_pct']:+.1f} % at M = {cfg.bluntness.report_mach:g}, "
                       f"apogee {b['d_apogee']:+.0f} m")
    if heat:
        out += ["", "## Heating (nominal / envelope, degC)", ""]
        out.append(f"* Stagnation temperature T_0: {heat['T0_peak'][0]:.0f} / {heat['T0_peak'][1]:.0f} "
                   f"(M_max {heat['mach_max'][0]:.2f} / {heat['mach_max'][1]:.2f})")
        out.append(f"* Metal tip insert: {heat['tip_insert'][0]:.0f} / {heat['tip_insert'][1]:.0f}")
        out.append(f"* Printed tip surface: {heat['printed_tip'][0]:.0f} / {heat['printed_tip'][1]:.0f}")
        out.append(f"* Shell inner face (max over stations): {heat['shell_inner'][0]:.0f} / {heat['shell_inner'][1]:.0f}")
    if design:
        out += ["", "## Selected design", ""]
        out.append(f"* {design['profile']}, L = {1e3 * design['L_m']:.0f} mm (L/D = {design['fineness']:g}), "
                   f"physical length {1e3 * design['L_physical_m']:.1f} mm, tip radius {1e3 * design['tip_radius_m']:g} mm")
        out.append(f"* Volume {1e3 * design['volume_m3']:.2f} L, wetted area {design['wetted_area_m2']:.3f} m^2, "
                   f"CP {1e3 * design['x_cp_m']:.0f} mm from the tip, shoulder angle {design['shoulder_angle_deg']:.1f} deg")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------- LaTeX
def _name(st: Study, s: str, table: dict) -> str:
    return table.get(s, st.profiles[s].label)


def tab_trade(st: Study, t: pd.DataFrame) -> str:
    """LaTeX table of the candidates at the baseline and design fineness."""
    cfg = st.cfg
    fb, fd = cfg.nose.baseline_fineness, cfg.nose.design_fineness
    sel = t[t.fineness.isin(sorted({fb, fd}))].copy()
    order = list(cfg.nose.profiles)
    sel["o"] = sel["shape"].map(order.index)
    sel = sel.sort_values(["fineness", "o"])
    M3 = cfg.aerodynamics.machs[:3]
    pl = cfg.payload.name
    tip = cfg.heating.materials[cfg.heating.tip_material].get("short", cfg.heating.materials[cfg.heating.tip_material].get("label", ""))
    rq = 1e3 * cfg.payload.required_radius
    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{Candidates at $L/D={fb:g}$ ({cfg.nose.baseline_label}, \SI{{{1e3 * st.L(fb):.0f}}}{{mm}}) and "
             rf"$L/D={fd:g}$ (\SI{{{1e3 * st.L(fd):.0f}}}{{mm}}). "
             rf"$\Delta z$: apogee change vs.\ the baseline ({st.profiles[cfg.nose.baseline_profile].label}, $L/D={fb:g}$); "
             rf"mass: shell + {tip} tip; {pl} in nose: curved length with $r\geq\SI{{{rq:.0f}}}{{mm}}$; bay: curved length "
             rf"plus the cylinder the {pl} still needs.}}",
             r"\label{tab:trade}", r"\small",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
             rf"$L/D$ & Profile & \multicolumn{{{len(M3)}}}{{c}}{{$C_{{D,\rm w}}$ at $M=$}} & $\Delta z$ & mass & {pl} in nose & bay & $\Delta$SM \\",
             r" & & " + " & ".join(f"{M:g}" for M in M3) + r" & [m] & [g] & [mm] & [mm] & [cal] \\", r"\midrule"]
    last = None
    for _, r in sel.iterrows():
        if last is not None and r.fineness != last:
            lines.append(r"\midrule")
        last = r.fineness
        name = st.profiles[r["shape"]].label
        name = rf"\textbf{{{name}}}" if r["shape"] == cfg.nose.design_profile else name
        cds = " & ".join(f"{r[f'cd_w_{mach_key(M)}']:.4f}" for M in M3)
        lines.append(rf"{r.fineness:g} & {name} & {cds} & "
                     rf"{r.d_apogee:+.0f} & {r.mass_g:.0f} & {r.l3U_mm:.0f} & {r.bay_mm:.0f} & {r.dSM_cal:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_matrix(st: Study, m: pd.DataFrame) -> str:
    """LaTeX table of the weighted decision matrix."""
    cfg = st.cfg
    order = list(m.columns)
    rows = []
    weights = matrix_weights(cfg.trade_matrix.weights)
    for c in weights:
        name = CRITERIA[c].format(payload=cfg.payload.name)
        rows.append(f"{name} ({weights[c]:.2f}) & " + " & ".join(f"{x:.1f}" for x in m.loc[c].values) + r" \\")
    tot = m.loc["total"].values
    best = int(np.nanargmax(tot))
    cells = [rf"\textbf{{{x:.2f}}}" if i == best else f"{x:.2f}" for i, x in enumerate(tot)]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Weighted total} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{table}[H]", r"\centering",
            rf"\caption{{Trade-off matrix at $L/D = {cfg.trade_matrix.fineness:g}$ (1 = worst, 5 = best). Performance: apogee at "
            r"equal length (Table~\ref{tab:trade}). " + rf"Integration: {cfg.payload.name} length inside the curved part. "
            r"Manufacture: tangent shoulder, mould and tip. "
            r"Stability: change in static margin.}",
            r"\label{tab:matrix}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}l" + "c" * len(order) + r"@{}}", r"\toprule",
            r"Criterion (weight) & " + " & ".join(_name(st, s, MATRIX_NAMES) for s in order) + r" \\", r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(head + rows + tail) + "\n"


def _direct_rows(st: Study) -> pd.DataFrame:
    """CFD rows computed directly at the design fineness (studies other than the first one)."""
    cfg = st.cfg
    d = aero.select_from_config(st.db, cfg, rn=0.0)
    return d[np.isclose(d.fineness, cfg.nose.design_fineness)]


def tab_catalogue(st: Study, t: pd.DataFrame) -> str:
    """LaTeX table of every profile at the design fineness."""
    cfg = st.cfg
    f = cfg.nose.design_fineness
    order = list(cfg.nose.profiles)
    cat = st.catalogue(t, tuple(x for x in sorted(set(t.fineness)) if x >= cfg.nose.baseline_fineness - 1e-9))
    rows = []

    def add(label, vals, fmt):
        rows.append(label + " & " + " & ".join(fmt.format(v) if v == v else "--" for v in vals) + r" \\")

    add(r"Volume [L]", cat.loc["volume_L", order], "{:.2f}")
    add(r"Wetted area [m$^2$]", cat.loc["wetted_area_m2", order], "{:.3f}")
    add(r"Tip semi-angle [$^\circ$]", cat.loc["tip_semi_angle_deg", order], "{:.0f}")
    add(r"Shoulder angle [$^\circ$]", cat.loc["shoulder_angle_deg", order], "{:.1f}")
    add(r"$x_{cp}/L$ (Barrowman)", cat.loc["xcp_over_L", order], "{:.3f}")
    rows.append(r"\midrule")
    for idx in [i for i in cat.index if i.startswith("cd_w_direct_")]:
        M = float(_mach_from_key(idx[len("cd_w_direct_"):]))
        add(rf"$C_{{D,\rm w}}$ at $M={M}$", cat.loc[idx, order], "{:.4f}")
    rows.append(r"\midrule")
    for idx in [i for i in cat.index if i.startswith("d_apogee_LD")]:
        add(rf"$\Delta z$ at $L/D={idx[len('d_apogee_LD'):]}$ [m]", cat.loc[idx, order], "{:+.0f}")
    rows.append(r"\midrule")
    add(r"Shell + tip mass [g]", cat.loc["mass_g", order], "{:.0f}")
    add(rf"{cfg.payload.name} inside nose [mm]", cat.loc["l3U_mm", order], "{:.0f}")
    add(r"Nose-bay length [mm]", cat.loc["bay_mm", order], "{:.0f}")
    add(r"$\Delta$ static margin [cal]", cat.loc["dSM_cal", order], "{:+.2f}")
    fb = cfg.nose.baseline_fineness
    head = [r"\begin{table}[H]", r"\centering",
            rf"\caption{{Design catalogue at $L/D = {f:g}$ ($L = \SI{{{1e3 * st.L(f):.0f}}}{{mm}}$). Wave drag from the direct "
            rf"Euler runs at this fineness; $\Delta z$ relative to {st.profiles[cfg.nose.baseline_profile].label} at $L/D={fb:g}$ "
            rf"({cfg.nose.baseline_label}), including length and mass effects.}}",
            r"\label{tab:catalogue}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}l" + "c" * len(order) + r"@{}}", r"\toprule",
            r" & " + " & ".join(_name(st, s, CATALOGUE_NAMES) for s in order) + r" \\",
            r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(head + rows + tail) + "\n"


def _mach_from_key(k: str) -> str:
    return k[0] + "." + k[1:] if len(k) > 1 else k


def tab_check(st: Study) -> Optional[str]:
    """Interpolated (from the first study) against direct wave drag at the design fineness."""
    cfg = st.cfg
    f = cfg.nose.design_fineness
    full = st.db
    s0 = cfg.aerodynamics.studies[0]
    interp = full[(full.study == s0) & (full.rn == 0) & (full.fineness <= cfg.aerodynamics.max_fineness)]
    direct = full[(full.study != s0) & full.study.isin(cfg.aerodynamics.studies) & (full.rn == 0)
                  & np.isclose(full.fineness, f)]
    if direct.empty:
        return None
    machs = sorted(set(direct.mach))
    rows = []
    for s in cfg.nose.profiles:
        try:
            w = aero.wave_drag(cfg, interp, s, f)
        except ValueError:
            continue
        fsd = w.fineness_data
        if len(fsd) == 1 or not fsd[0] <= f <= fsd[-1]:
            near = fsd[np.argmin(abs(fsd - f))]
            how = f"scaled from {near:g}"
        else:
            how = f"interp. {fsd[0]:g}--{fsd[-1]:g}"
        cells = []
        for M in machs:
            dd = direct[(direct["shape"] == s) & (direct.mach == M)].cd
            if dd.empty:
                cells.append("-- & -- & --")
                continue
            di = float(dd.values[0])
            it = float(w(M))
            cells.append(f"{it:.4f} & {di:.4f} & {100 * (it / di - 1):+.1f}")
        rows.append(f"{st.profiles[s].label} & {how} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{table}[H]", r"\centering",
            rf"\caption{{Wave drag at $L/D = {f:g}$: value used before the direct runs (interpolated in "
            r"$C_{D,\rm w}(L/D)^2$, or scaled with $1/(L/D)^2$ from a single fineness) against the direct "
            r"Euler result, and the error of the former.}",
            r"\label{tab:check45}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}ll" + "ccr" * len(machs) + r"@{}}", r"\toprule",
            r" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$M={M:g}$}}" for M in machs) + r" \\",
            r"Shape & source & " + " & ".join(r"interp. & direct & [\%]" for _ in machs) + r" \\",
            r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(head + rows + tail) + "\n"


def macros(st: Study, t: pd.DataFrame, blunt: list, heat: dict, design: dict) -> str:
    """LaTeX macros of the numbers quoted in the design report."""
    cfg = st.cfg

    def row(s, f):
        sel = t[(t["shape"] == s) & (np.isclose(t.fineness, f))]
        return sel.iloc[0] if len(sel) else None
    m = {}
    bp, bf = st.baseline
    dp, df_, rnr = st.design
    b_row, d_row = row(bp, bf), row(dp, df_)
    if b_row is not None and row("ogive", bf) is not None:
        m["TRADEOGIVE"] = rf"\SI{{{b_row.apogee - row('ogive', bf).apogee:.0f}}}{{m}}"
    if b_row is not None and row("power050", bf) is not None:
        m["TRADEPOWER"] = rf"\SI{{{b_row.apogee - row('power050', bf).apogee:.0f}}}{{m}}"
    lr = row(bp, df_)
    if b_row is not None and lr is not None:
        m["TRADELEN"] = rf"\SI{{{lr.apogee - b_row.apogee:.0f}}}{{m}}"
    keys = [("BLUNTCOST", "BLUNTAPOA"), ("BLUNTTEN", "BLUNTAPOB")]
    for (kc, kz), b in zip(keys, blunt or []):
        m[kc] = rf"${b['drag_cost_pct']:+.1f}\,\%$"
        m[kz] = rf"\SI{{{b['d_apogee']:+.0f}}}{{m}}"
    if heat:
        tip, pet, sh = heat["tip_insert"], heat["printed_tip"], heat["shell_inner"]
        m["TIPAL"] = rf"\SI{{{np.ceil(tip[1]):.0f}}}{{\celsius}}"
        m["PETNOM"] = rf"\SI{{{pet[0]:.0f}}}{{\celsius}}"
        m["PETENV"] = rf"\SI{{{pet[1]:.0f}}}{{\celsius}}"
        m["TIPALNOM"] = rf"\SI{{{np.ceil(tip[0]):.0f}}}{{\celsius}}"
        m["SHELLNOM"] = rf"\SI{{{sh[0]:.0f}}}{{\celsius}}"
        m["SHELLENV"] = rf"\SI{{{sh[1]:.0f}}}{{\celsius}}"
    L = design["L_m"]
    m["FINL"] = rf"\SI{{{1e3 * L:.0f}}}{{mm}}"
    m["FINLPHYS"] = rf"\SI{{{1e3 * design['L_physical_m']:.1f}}}{{mm}}"
    m["FINTRUNC"] = rf"\SI{{{1e3 * design['truncation_m']:.1f}}}{{mm}}"
    m["FINXT"] = rf"\SI{{{1e3 * design['tangency_x_m']:.1f}}}{{mm}}"
    m["FINVOL"] = rf"\SI{{{1e3 * design['volume_m3']:.2f}}}{{L}}"
    m["FINAREA"] = rf"\SI{{{design['wetted_area_m2']:.3f}}}{{m^2}}"
    if d_row is not None:
        m["FINMASS"] = rf"\SI{{{d_row.mass_g:.0f}}}{{g}}"
    m["FINXCP"] = rf"\SI{{{1e3 * design['x_cp_m']:.0f}}}{{mm}}"
    if d_row is not None:
        if "l3U_mm" in d_row:
            m["FINTHREEU"] = rf"\SI{{{d_row.l3U_mm:.0f}}}{{mm}}"
            m["FINBAY"] = rf"\SI{{{d_row.bay_mm:.0f}}}{{mm}}"
            m["FINCYL"] = rf"\SI{{{d_row.bay_mm - 1e3 * L:.0f}}}{{mm}}"
        m["FINDAPO"] = rf"\SI{{{d_row.d_apogee:+.0f}}}{{m}}"
        m["FINAPO"] = rf"\SI{{{d_row.apogee:.0f}}}{{m}}"
        m["FINDSM"] = rf"{d_row.dSM_cal:+.2f}"
        for key, M in zip(["FINCDA", "FINCDB", "FINCDC", "FINCDD", "FINCDE", "FINCDF"], cfg.aerodynamics.machs):
            m[key] = f"{d_row[f'cd_w_{mach_key(M)}']:.4f}"
        m["FINMMAX"] = f"{d_row.mach_max:.2f}"
    return "\n".join(rf"\newcommand{{\{k_}}}{{{v}}}" for k_, v in m.items()) + "\n"


def write_latex(st: Study, outdir: Path, t: pd.DataFrame, matrix: Optional[pd.DataFrame], blunt: list,
                heat: Optional[dict], design: dict) -> list[Path]:
    """Write tab_trade.tex, tab_matrix.tex, tab_catalogue.tex, tab_check45.tex and macros.tex."""
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    if st.cfg.payload.enabled:
        files["tab_trade.tex"] = tab_trade(st, t)
    if matrix is not None:
        files["tab_matrix.tex"] = tab_matrix(st, matrix)
    if st.cfg.payload.enabled:
        files["tab_catalogue.tex"] = tab_catalogue(st, t)
    chk = tab_check(st)
    if chk:
        files["tab_check45.tex"] = chk
    files["macros.tex"] = macros(st, t, blunt, heat, design)
    out = []
    for name, text in files.items():
        p = outdir / name
        p.write_text(text, encoding="utf-8")
        out.append(p)
    return out

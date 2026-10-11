"""Generate the LaTeX tables and the numbers quoted in the text from the results files."""
import os
import numpy as np
import pandas as pd

import analysis as an
import thermal as th
from flight import simulate
from figures import FINAL, SHORT

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEC = os.path.join(HERE, "..", "report", "sections")
R = an.R


def tab_trade(t):
    sel = t[t.fineness.isin([4.0, 4.5])].copy()
    order = list(SHORT)
    sel["o"] = sel["shape"].map(order.index)
    sel = sel.sort_values(["fineness", "o"])
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Candidates at $L/D=4$ (LVS length, \SI{720}{mm}) and $L/D=4.5$ (\SI{810}{mm}). "
             r"$\Delta z$: apogee change vs.\ the baseline (Von K\'arm\'an, $L/D=4$); "
             r"mass: shell + Al tip; 3U in nose: curved length with $r\geq\SI{77}{mm}$; bay: curved length plus the cylinder the 3U still needs.}",
             r"\label{tab:trade}", r"\small",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
             r"$L/D$ & Profile & \multicolumn{3}{c}{$C_{D,\rm w}$ at $M=$} & $\Delta z$ & mass & 3U in nose & bay & $\Delta$SM \\",
             r" & & 1.1 & 1.2 & 1.4 & [m] & [g] & [mm] & [mm] & [cal] \\", r"\midrule"]
    last = None
    for _, r in sel.iterrows():
        if last is not None and r.fineness != last:
            lines.append(r"\midrule")
        last = r.fineness
        bold = r["shape"] == FINAL["shape"]
        name = SHORT[r["shape"]]
        name = rf"\textbf{{{name}}}" if bold else name
        lines.append(rf"{r.fineness:g} & {name} & {r.cd_w_11:.4f} & {r.cd_w_12:.4f} & {r.cd_w_14:.4f} & "
                     rf"{r.d_apogee:+.0f} & {r.mass_g:.0f} & {r.l3U_mm:.0f} & {r.bay_mm:.0f} & {r.dSM_cal:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(SEC, "tab_trade.tex"), "w").write("\n".join(lines) + "\n")


def tab_matrix(t):
    """Weighted trade matrix at L/D = 4.5; scores scaled linearly between worst and best."""
    order = list(SHORT)
    d = t[np.isclose(t.fineness, 4.5)].set_index("shape").reindex(order)

    def lin(v, hi_good=True):
        v = np.asarray(v, float)
        s = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
        return 1 + 4 * (s if hi_good else 1 - s)

    # relative weights 4:2:2:1, normalised to add up to 1
    crit = [
        ("Performance", 4, lin(d.d_apogee)),
        ("Integration, 3U fit", 2, lin(d.l3U_mm)),
        # tangent shoulder 5; kink penalised; cone easiest mould
        ("Manufacture", 2, np.array([3, 5, 5, 3, 2, 5, 5], float)),
        ("Stability", 1, lin(d.dSM_cal)),
    ]
    wsum = sum(w for _, w, _ in crit)
    rows = []
    tot = np.zeros(len(order))
    for name, w, sc in crit:
        tot += w / wsum * sc
        rows.append(f"{name} ({w / wsum:.2f}) & " + " & ".join(f"{x:.1f}" for x in sc) + r" \\")
    best = int(np.nanargmax(tot))
    cells = [rf"\textbf{{{x:.2f}}}" if i == best else f"{x:.2f}" for i, x in enumerate(tot)]
    rows.append(r"\midrule")
    rows.append(r"\textbf{Weighted total} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{table}[H]", r"\centering",
            r"\caption{Trade-off matrix at $L/D = 4.5$ (1 = worst, 5 = best). Performance: apogee at "
            r"equal length (Table~\ref{tab:trade}). Integration: 3U length inside the curved part. "
            r"Manufacture: tangent shoulder, mould and tip. "
            r"Stability: change in static margin.}",
            r"\label{tab:matrix}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
            r"Criterion (weight) & Cone & Ogive & Parab. & Power $\tfrac12$ & Power $\tfrac34$ & LV-Haack & "
            r"Von K\'arm\'an \\", r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(SEC, "tab_matrix.tex"), "w").write("\n".join(head + rows + tail) + "\n")
    print("matrix totals", dict(zip(order, np.round(tot, 2))))


def tab_catalogue(t):
    """All metrics of the seven designs at L/D = 4.5 (direct Euler values)."""
    from geometry import volume_and_area, base_angle_deg, tip_half_angle_deg
    order = list(SHORT)
    L = 2 * R * 4.5
    d45 = an.load_sweep()
    d45 = d45[np.isclose(d45.fineness, 4.5)]
    tt = t.set_index(["shape", "fineness"])
    rows = []

    def add(label, vals, fmt):
        rows.append(label + " & " + " & ".join(fmt.format(v) if v == v else "--" for v in vals) + r" \\")

    V = [volume_and_area(s, L, R)[0] * 1e3 for s in order]
    S = [volume_and_area(s, L, R)[1] for s in order]
    add(r"Volume [L]", V, "{:.2f}")
    add(r"Wetted area [m$^2$]", S, "{:.3f}")
    add(r"Tip semi-angle [$^\circ$]", [tip_half_angle_deg(s, L, R) for s in order], "{:.0f}")
    add(r"Shoulder angle [$^\circ$]", [abs(base_angle_deg(s, L, R)) for s in order], "{:.1f}")
    add(r"$x_{cp}/L$ (Barrowman)", [tt.loc[(s, 4.5)].xcp_over_L for s in order], "{:.3f}")
    rows.append(r"\midrule")
    for M in (1.2, 1.4, 1.6):
        vals = []
        for s in order:
            v = d45[(d45["shape"] == s) & (d45.mach == M)].cd
            vals.append(float(v.values[0]) if len(v) else np.nan)
        add(rf"$C_{{D,\rm w}}$ at $M={M}$", vals, "{:.4f}")
    rows.append(r"\midrule")
    add(r"$\Delta z$ at $L/D=4$ [m]", [tt.loc[(s, 4.0)].d_apogee for s in order], "{:+.0f}")
    add(r"$\Delta z$ at $L/D=4.5$ [m]", [tt.loc[(s, 4.5)].d_apogee for s in order], "{:+.0f}")
    add(r"$\Delta z$ at $L/D=5$ [m]", [tt.loc[(s, 5.0)].d_apogee for s in order], "{:+.0f}")
    rows.append(r"\midrule")
    add(r"Shell + tip mass [g]", [tt.loc[(s, 4.5)].mass_g for s in order], "{:.0f}")
    add(r"3U inside nose [mm]", [tt.loc[(s, 4.5)].l3U_mm for s in order], "{:.0f}")
    add(r"Nose-bay length [mm]", [tt.loc[(s, 4.5)].bay_mm for s in order], "{:.0f}")
    add(r"$\Delta$ static margin [cal]", [tt.loc[(s, 4.5)].dSM_cal for s in order], "{:+.2f}")
    head = [r"\begin{table}[H]", r"\centering",
            r"\caption{Design catalogue at $L/D = 4.5$ ($L = \SI{810}{mm}$). Wave drag from the direct "
            r"Euler runs at this fineness; $\Delta z$ relative to Von K\'arm\'an at $L/D=4$ "
            r"(LVS length), including length and mass effects.}",
            r"\label{tab:catalogue}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
            r" & Cone & Ogive & Parabolic & Power $\tfrac12$ & Power $\tfrac34$ & LV-Haack & Von K\'arm\'an \\",
            r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(SEC, "tab_catalogue.tex"), "w").write("\n".join(head + rows + tail) + "\n")


def tab_check45():
    """Interpolated (from L/D = 3, 4, 5 or scaled from L/D = 4) vs direct Euler at L/D = 4.5."""
    full = pd.read_csv(os.path.join(RES, "euler_results.csv"))
    main = full[(full.study == "main") & (full.rn == 0) & (full.fineness <= 5)]
    direct = full[(full.study == "f45") & (full.rn == 0)]
    order = list(SHORT)
    rows = []
    for s in order:
        w = an.WaveDrag(main, s, 4.5)
        how = "interp. 3--5" if w.n_fineness >= 3 else ("scaled from 4" if w.n_fineness == 1 else "interp. 4--5")
        cells = []
        for M in (1.2, 1.4, 1.6):
            di = float(direct[(direct["shape"] == s) & (direct.mach == M)].cd.values[0])
            it = float(w(M))
            cells.append(f"{it:.4f} & {di:.4f} & {100 * (it / di - 1):+.1f}")
        rows.append(f"{SHORT[s]} & {how} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{table}[H]", r"\centering",
            r"\caption{Wave drag at $L/D = 4.5$: value used before the direct runs (interpolated in "
            r"$C_{D,\rm w}(L/D)^2$, or scaled with $1/(L/D)^2$ from a single fineness) against the direct "
            r"Euler result, and the error of the former.}",
            r"\label{tab:check45}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}llccrccrccr@{}}", r"\toprule",
            r" & & \multicolumn{3}{c}{$M=1.2$} & \multicolumn{3}{c}{$M=1.4$} & \multicolumn{3}{c}{$M=1.6$} \\",
            r"Shape & source & interp. & direct & [\%] & interp. & direct & [\%] & interp. & direct & [\%] \\",
            r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(SEC, "tab_check45.tex"), "w").write("\n".join(head + rows + tail) + "\n")


def macros(t, k):
    def row(s, f):
        return t[(t["shape"] == s) & (np.isclose(t.fineness, f))].iloc[0]
    m = {}
    vk4, vk45 = row("vonkarman", 4.0), row("vonkarman", 4.5)
    m["TRADEOGIVE"] = rf"\SI{{{vk4.apogee - row('ogive', 4.0).apogee:.0f}}}{{m}}"
    m["TRADEPOWER"] = rf"\SI{{{vk4.apogee - row('power050', 4.0).apogee:.0f}}}{{m}}"
    m["TRADELEN"] = rf"\SI{{{vk45.apogee - vk4.apogee:.0f}}}{{m}}"
    # bluntness cost (Von Karman, L/D = 4): wave drag at M = 1.4 and apogee effect of the
    # Mach-dependent drag ratio applied to the baseline nose
    d = an.load_sweep("blunt")
    d0 = an.load_sweep("main")
    base = d0[(d0["shape"] == "vonkarman") & (d0.fineness == 4.0)].set_index("mach").cd
    vehb = an.make_vehicle(an.load_sweep(), *an.BASELINE, k)
    z0 = simulate(vehb, mass_model="isp")["apogee"]
    for key, rnr, zkey in [("BLUNTCOST", 0.05, "BLUNTAPOA"), ("BLUNTTEN", 0.1, "BLUNTAPOB")]:
        b = d[(d["shape"] == "vonkarman") & np.isclose(d.rn, rnr)].set_index("mach").cd
        Ms = np.array(sorted(set(b.index) & set(base.index)))
        ratio = np.array([b.loc[M] / base.loc[M] for M in Ms])
        m[key] = rf"${100 * (b.loc[1.4] / base.loc[1.4] - 1):+.1f}\,\%$"
        vb = an.make_vehicle(an.load_sweep(), *an.BASELINE, k)
        cdw0 = vb.cdw
        vb.cdw = lambda M, c=cdw0, x=Ms, r=ratio: c(M) * np.interp(M, x, r)
        m[zkey] = rf"\SI{{{simulate(vb, mass_model='isp')['apogee'] - z0:+.0f}}}{{m}}"
    # heating along the nominal and envelope trajectories
    veh = an.make_vehicle(an.load_sweep(), *an.BASELINE, k)
    nom = simulate(veh, mass_model="isp", dt_out=0.02)
    env = simulate(veh, mass_model="lvo", T_ground=303.15, dt_out=0.02)
    L = 2 * R * FINAL["f"]
    rn = FINAL["rn_over_R"] * R
    tip = [th.tip_insert(tr, FINAL["shape"], L, R, rn, 0.08, an.M_TIP_AL)["T"].max() - 273.15 for tr in (nom, env)]
    m["TIPAL"] = rf"\SI{{{np.ceil(tip[1]):.0f}}}{{\celsius}}"
    pet = [th.wall_response(tr, "PET-CF (FDM)", 0.004, "stagnation", rn=rn)["T_out"].max() - 273.15
           for tr in (nom, env)]
    m["PETNOM"] = rf"\SI{{{pet[0]:.0f}}}{{\celsius}}"
    m["PETENV"] = rf"\SI{{{pet[1]:.0f}}}{{\celsius}}"
    m["TIPALNOM"] = rf"\SI{{{np.ceil(tip[0]):.0f}}}{{\celsius}}"
    sh = []
    for tr in (nom, env):
        sh.append(max(th.wall_response(tr, "CFRP (through-thickness)", 0.0012, x)["T_in"].max()
                      for x in (0.05, 0.5 * L, L)) - 273.15)
    m["SHELLNOM"] = rf"\SI{{{sh[0]:.0f}}}{{\celsius}}"
    m["SHELLENV"] = rf"\SI{{{sh[1]:.0f}}}{{\celsius}}"
    # final design definition
    from geometry import blunt_tangent_point, volume_and_area, base_angle_deg
    fin = row(FINAL["shape"], FINAL["f"])
    xt, xc = blunt_tangent_point(FINAL["shape"], L, R, rn)
    L_phys = L - (xc - rn)
    V, S = volume_and_area(FINAL["shape"], L, R, rn)
    m["FINL"] = rf"\SI{{{1e3 * L:.0f}}}{{mm}}"
    m["FINLPHYS"] = rf"\SI{{{1e3 * L_phys:.1f}}}{{mm}}"
    m["FINTRUNC"] = rf"\SI{{{1e3 * (xc - rn):.1f}}}{{mm}}"
    m["FINXT"] = rf"\SI{{{1e3 * xt:.1f}}}{{mm}}"
    m["FINVOL"] = rf"\SI{{{1e3 * V:.2f}}}{{L}}"
    m["FINAREA"] = rf"\SI{{{S:.3f}}}{{m^2}}"
    m["FINMASS"] = rf"\SI{{{fin.mass_g:.0f}}}{{g}}"
    m["FINXCP"] = rf"\SI{{{1e3 * (L - V / (np.pi * R**2)):.0f}}}{{mm}}"
    m["FINTHREEU"] = rf"\SI{{{fin.l3U_mm:.0f}}}{{mm}}"
    m["FINBAY"] = rf"\SI{{{fin.bay_mm:.0f}}}{{mm}}"
    m["FINCYL"] = rf"\SI{{{fin.bay_mm - 1e3 * L:.0f}}}{{mm}}"
    m["FINDAPO"] = rf"\SI{{{fin.d_apogee:+.0f}}}{{m}}"
    m["FINAPO"] = rf"\SI{{{fin.apogee:.0f}}}{{m}}"
    m["FINDSM"] = rf"{fin.dSM_cal:+.2f}"
    m["FINCDA"] = f"{fin.cd_w_11:.4f}"
    m["FINCDB"] = f"{fin.cd_w_12:.4f}"
    m["FINCDC"] = f"{fin.cd_w_14:.4f}"
    m["FINCDD"] = f"{fin.cd_w_16:.4f}"
    m["FINMMAX"] = f"{fin.mach_max:.2f}"
    out = "\n".join(rf"\newcommand{{\{k_}}}{{{v}}}" for k_, v in m.items())
    open(os.path.join(SEC, "macros.tex"), "w").write(out + "\n")
    print(out)
    print("tip", tip, "shell", sh, "nominal Mmax", nom["M"].max(), "env Mmax", env["M"].max())


if __name__ == "__main__":
    t = pd.read_csv(os.path.join(RES, "trade_table.csv"))
    k = float(open(os.path.join(RES, "calibration.txt")).readline().split("=")[1])
    tab_trade(t)
    tab_matrix(t)
    tab_catalogue(t)
    tab_check45()
    macros(t, k)

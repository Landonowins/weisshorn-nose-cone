"""Design-catalogue figures: to-scale gallery, Euler flow fields, drag breakdown, robustness.

usage: python figures_extra.py [gallery fields breakdown robustness]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import COLORS, STYLES, MARKERS, INK, INK2, GRID  # noqa: E402
from geometry import SHAPES, surface, base_angle_deg  # noqa: E402
import analysis as an  # noqa: E402
from flight import LVO  # noqa: E402
from figures import SHORT, FINAL, FIG, RES  # noqa: E402

R = an.R
ORDER = list(SHAPES)
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
EQ = {
    "cone": r"$\eta=\xi$",
    "ogive": r"circular arc, $\rho=(R^2+L^2)/2R$",
    "parabolic": r"$\eta=2\xi-\xi^2$",
    "power050": r"$\eta=\xi^{1/2}$",
    "power075": r"$\eta=\xi^{3/4}$",
    "lvhaack": r"Haack, $C=\tfrac13$",
    "vonkarman": r"Haack, $C=0$",
}


def shaded_body(ax, shape, L, R, sh, color, rn=0.0, tip_x=None, res=0.6):
    """Side view of the solid of revolution with Lambert + Phong shading (to scale, mm)."""
    from matplotlib.colors import to_rgb
    xs = np.arange(0, 1e3 * (L + sh) + res, res)
    ys = np.arange(-1e3 * R - res, 1e3 * R + res, res)
    xb, yb = surface(shape, L, R, rn=rn, n=3000)
    x0 = xb[0]
    xb = 1e3 * (xb - x0)
    yb = 1e3 * yb
    r = np.interp(xs, xb, yb, left=0.0, right=1e3 * R)
    dr = np.gradient(r, xs)
    Xg, Yg = np.meshgrid(xs, ys)
    rg = r[None, :]
    inside = np.abs(Yg) < rg
    sphi = np.clip(Yg / np.maximum(rg, 1e-9), -1, 1)
    cphi = np.sqrt(1 - sphi**2)
    n = np.stack([-np.broadcast_to(dr[None, :], Yg.shape), sphi, cphi], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Ld = np.array([-0.35, 0.55, 0.76])
    Ld /= np.linalg.norm(Ld)
    H = Ld + np.array([0, 0, 1.0])
    H /= np.linalg.norm(H)
    diff = np.clip(n @ Ld, 0, 1)
    spec = np.clip(n @ H, 0, 1) ** 40
    base = np.array(to_rgb(color)) * 0.65 + 0.35
    img = np.ones(Yg.shape + (4,))
    rgb = base[None, None, :] * (0.32 + 0.68 * diff[..., None]) + 0.35 * spec[..., None]
    if tip_x is not None:
        grey = np.array(to_rgb("#a7b0b8"))
        mask = Xg <= tip_x
        rgb[mask] = grey * (0.32 + 0.68 * diff[mask][:, None]) + 0.45 * spec[mask][:, None]
    img[..., :3] = np.clip(rgb, 0, 1)
    img[..., 3] = inside.astype(float)
    ax.imshow(img, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", interpolation="bilinear")
    X = np.concatenate([xb, [1e3 * (L + sh) - 1e3 * x0 * 0, 1e3 * (L + sh)], ])
    ax.plot(np.concatenate([xb, [1e3 * (L + sh) - 1e3 * x0]]), np.concatenate([yb, [1e3 * R]]), color=INK, lw=0.4)
    ax.plot(np.concatenate([xb, [1e3 * (L + sh) - 1e3 * x0]]), -np.concatenate([yb, [1e3 * R]]), color=INK, lw=0.4)
    ax.plot([1e3 * (L + sh) - 1e3 * x0] * 2, [-1e3 * R, 1e3 * R], color=INK, lw=0.4)
    ax.plot([1e3 * (L - x0)] * 2, [-1e3 * R, 1e3 * R], color=INK2, lw=0.35, ls=(0, (2, 2)))
    _ = X


def fig_gallery():
    """All seven noses at L/D = 4.5, to scale, with the 3U envelope and the key numbers."""
    t = pd.read_csv(os.path.join(RES, "trade_table.csv"))
    t = t[np.isclose(t.fineness, 4.5)].set_index("shape")
    L = 2 * R * 4.5
    sh = 0.060
    fig, axs = plt.subplots(len(ORDER), 2, figsize=(fs.WIDTH, 1.02 * fs.WIDTH),
                            gridspec_kw=dict(width_ratios=[2.15, 1], hspace=0.12, wspace=0.02))
    for row, s in enumerate(ORDER):
        a, b = axs[row]
        shaded_body(a, s, L, R, sh, COLORS[s])
        # 3U unit, pushed forward as far as the profile allows (half-diagonal + clearance)
        l3 = 1e3 * an.L_3U
        x3_front = 1e3 * (L - t.loc[s, "l3U_mm"] / 1e3)
        a.add_patch(plt.Rectangle((x3_front, -50), l3, 100, fill=False, ec=INK, lw=0.6, ls=(0, (3, 2))))
        x_end = max(1e3 * (L + sh), x3_front + l3 + 10)
        if x_end > 1e3 * (L + sh):   # payload-bay cylinder needed behind the shoulder
            for yy in (90, -90):
                a.plot([1e3 * (L + sh), x_end], [yy, yy], color=INK2, lw=0.5, ls=(0, (2, 2)))
            a.plot([x_end, x_end], [-90, 90], color=INK2, lw=0.5, ls=(0, (2, 2)))
        a.set_xlim(-10, 1e3 * (L + sh) + 300)
        a.set_ylim(-100, 122)
        a.set_aspect("equal")
        a.axis("off")
        title = SHORT[s] + (" (recommended)" if s == FINAL["shape"] else "")
        a.text(-8, 96, rf"\textbf{{{title}}}", fontsize=8, ha="left", va="bottom",
               color=COLORS[s] if s == FINAL["shape"] else INK)
        r = t.loc[s]
        txt = (rf"{EQ[s]}" "\n"
               rf"$C_{{D,\mathrm{{w}}}}(1.4)={r.cd_w_14:.4f}$, $\Delta z={r.d_apogee:+.0f}$ m" "\n"
               rf"mass {r.mass_g:.0f} g, 3U in nose {r.l3U_mm:.0f} mm" "\n"
               rf"shoulder ${base_angle_deg(s, L, R):.1f}^\circ$, $x_{{cp}}/L={r.xcp_over_L:.2f}$")
        b.text(0.0, 0.5, txt, fontsize=6.8, va="center", ha="left", transform=b.transAxes, color=INK,
               linespacing=1.35)
        b.axis("off")
    fs.save(fig, "fig_gallery", FIG)


def fig_fields(M=1.4):
    """Mach-number field around every nose at L/D = 4.5 (Euler)."""
    fig, axs = plt.subplots(4, 2, figsize=(fs.WIDTH, 0.80 * fs.WIDTH))
    axs = axs.ravel()
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.45)
    DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f4a3a2', '#c9302f'])
    pc = None
    for k, s in enumerate(ORDER):
        ax = axs[k]
        fn = os.path.join(RES, "fields", f"{s}_rn0_M{M:g}.npz")
        if not os.path.exists(fn):
            ax.axis("off")
            continue
        z = np.load(fn)
        X, Y, W = z["X"], z["Y"], z["W"].astype(float)
        c = np.sqrt(1.4 * W[..., 3] / W[..., 0])
        Mach = np.hypot(W[..., 1], W[..., 2]) / c
        Cp = (W[..., 3] - 1 / 1.4) / (0.5 * M**2)
        for sgn in (1, -1):
            pc = ax.pcolormesh(X, sgn * Y, Cp, cmap=DIV, norm=norm, shading="flat", rasterized=True)
        xb, yb = X[:, 0], Y[:, 0]
        ax.fill(np.concatenate([xb, xb[::-1]]), np.concatenate([yb, -yb[::-1]]), color="#d9d8d4", lw=0)
        ax.plot(xb, yb, color=INK, lw=0.5)
        ax.plot(xb, -yb, color=INK, lw=0.5)
        xc = 0.25 * (X[1:, 1:] + X[:-1, 1:] + X[1:, :-1] + X[:-1, :-1])
        yc = 0.25 * (Y[1:, 1:] + Y[:-1, 1:] + Y[1:, :-1] + Y[:-1, :-1])
        ax.tricontour(xc.ravel(), yc.ravel(), Mach.ravel(), levels=[1.0], colors=[INK], linewidths=0.5,
                      linestyles=[(0, (2, 1.5))])
        ax.set_xlim(-1.5, 13)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.text(0.015, 0.94, SHORT[s], transform=ax.transAxes, fontsize=7.5, va="top", color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    ax = axs[-1]
    ax.axis("off")
    cax = ax.inset_axes([0.08, 0.62, 0.84, 0.11])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(rf"pressure coefficient $C_p$ ($M_\infty = {M}$, $L/D = 4.5$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.text(0.5, 0.18, "red: compression, blue: expansion\ndashed: sonic line ($M=1$)",
            transform=ax.transAxes, ha="center", va="center", fontsize=7, color=INK2)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fs.save(fig, f"fig_fields_M{M:g}".replace(".", ""), FIG)


def fig_breakdown(k):
    """Vehicle drag build-up versus Mach for the recommended nose (coasting)."""
    df = an.load_sweep()
    veh = an.make_vehicle(df, FINAL["shape"], FINAL["f"], k)
    Ms = np.linspace(0.1, 1.6, 300)
    from flight import atmosphere
    comp = {key: [] for key in ["friction", "base", "fins", "excr", "nose_wave"]}
    for M in Ms:
        T, p, rho, mu, a = atmosphere(3000.0)
        c = veh.components(M, rho * M * a / mu, False)
        for key in comp:
            comp[key].append(c[key] * (k if key != "nose_wave" else 1.0))
    labels = {"friction": "skin friction (body + fins)", "base": "base", "fins": "fin pressure",
              "excr": "excrescences", "nose_wave": r"nose wave drag (Von K\'arm\'an, $L/D=4.5$)"}
    cols = {"friction": "#2a78d6", "base": "#eb6834", "fins": "#1baf7a", "excr": "#eda100", "nose_wave": "#4a3aa7"}
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.40 * fs.WIDTH))
    ys = [np.array(comp[key]) for key in comp]
    ax.stackplot(Ms, ys, colors=[cols[key] for key in comp], labels=[labels[key] for key in comp],
                 edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.set(xlim=(0.1, 1.6), ylim=(0, None), xlabel=r"$M$", ylabel=r"$C_D$ (coasting, $z = 3$ km)")
    ax.legend(loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0.0, 1.01), columnspacing=1.0)
    tot = np.sum(ys, axis=0)
    i = np.argmin(abs(Ms - 1.4))
    ax.annotate(rf"nose: {100 * ys[-1][i] / tot[i]:.0f}\,\% of $C_D$ at $M=1.4$", xy=(1.4, tot[i]),
                xytext=(1.15, tot.max() * 1.05), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    ax.set_ylim(0, tot.max() * 1.15)
    fs.save(fig, "fig_breakdown", FIG)


def fig_robustness():
    d = pd.read_csv(os.path.join(RES, "robustness.csv"))
    scen = list(dict.fromkeys(d.scenario))
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.46 * fs.WIDTH))
    marks = ["o", "v", "^", "s", "D", "P", "X"]
    seq = ["#0b0b0b", "#9ec5f4", "#256abf", "#86b6ef", "#1c5cab", "#eb6834", "#d03b3b"]
    for j, sc in enumerate(scen):
        dd = d[d.scenario == sc].set_index("shape").reindex(ORDER)
        ypos = np.arange(len(ORDER)) + (j - (len(scen) - 1) / 2) * 0.09
        ax.plot(dd.dz_vs_vk, ypos, ls="none", marker=marks[j], ms=4.2 if j else 5,
                mfc=seq[j] if j == 0 else "white", mec=seq[j], mew=1.0,
                label=sc.replace("%", r"\%").replace("30 C", r"30$^\circ$C"))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvspan(-30, 30, color=GRID, alpha=0.4, lw=0)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT[s] for s in ORDER])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.invert_yaxis()
    ax.set_xlabel(r"apogee relative to Von K\'arm\'an at $L/D=4.5$ [m]")
    ax.legend(loc="lower left", fontsize=6.8, ncol=1, bbox_to_anchor=(0.0, 0.0), frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.text(0, -0.75, r"$\pm30$ m", ha="center", fontsize=7, color=INK2)
    fs.save(fig, "fig_robustness", FIG)


def fig_grid():
    """The computational grid (coarse version, for explanation) with its boundaries labelled."""
    import euler_axi as e
    X, Y, ish = e.make_grid("vonkarman", 4.5, ni_nose=60, ni_cyl=20, nj=40, wall_dy=4e-3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.WIDTH, 0.42 * fs.WIDTH),
                               gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax in (a, b):
        ax.plot(X, Y, color="#9ec5f4", lw=0.35)
        ax.plot(X.T, Y.T, color="#9ec5f4", lw=0.35)
        xb, yb = X[:, 0], Y[:, 0]
        ax.fill_between(xb, 0, yb, color="#d9d8d4", lw=0)
        ax.plot(xb, yb, color=INK, lw=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    a.plot(X[:, -1], Y[:, -1], color="#eb6834", lw=1.4)
    a.plot(X[0, :], Y[0, :], color="#1baf7a", lw=1.4)
    a.plot(X[-1, :], Y[-1, :], color="#4a3aa7", lw=1.4)
    a.set_xlim(X.min() - 0.5, X.max() + 0.5)
    a.set_ylim(-0.3, 22)
    a.text(-4.0, 9, "far field\n(free stream\ncomes in)", color="#eb6834", fontsize=7, ha="center")
    a.text(-1.8, 0.5, "axis", color="#1baf7a", fontsize=7, ha="right")
    a.text(X.max() - 0.3, 12, "outflow", color="#4a3aa7", fontsize=7, ha="right", rotation=90)
    a.annotate("nose + cylinder\n(slip wall)", xy=(6, 0.9), xytext=(8, 6), fontsize=7,
               arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    fs.panel(a, "a", x=0.02, y=0.96)
    b.set_xlim(-0.4, 2.0)
    b.set_ylim(-0.02, 1.0)
    b.text(1.0, 0.12, "body", fontsize=7, ha="center")
    b.text(0.95, 0.9, "each quadrilateral\n= one cell", fontsize=7, ha="center", va="top",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    fs.panel(b, "b", x=0.03, y=0.96)
    fig.tight_layout(w_pad=1.0)
    fs.save(fig, "fig_grid", FIG)


def fig_schematics(fd=2.0):
    """Annotated schematics of the seven profiles, drawn at L/D = fd so that the differences show.

    (a) defines the notation; (b)-(h) show how each profile is built. Top line of each panel:
    the shoulder (tangent or kink); bottom line: the generator and the tip; right: the tip
    region magnified.
    """
    from matplotlib.patches import Arc, Rectangle
    from geometry import radius, slope
    Rs, Ls, stub = 1.0, 2.0 * fd, 0.9           # R = 1; L = 2 fd R; body tube drawn behind
    XL, YL = (-0.50, Ls + stub + 2.10), (-1.78, 1.78)
    xc = 0.5 * (XL[0] + XL[1])                   # centre of the panel, for the bottom line
    zx = 0.30                                    # tip window: -0.1 zx <= x <= zx (square)
    win = (-0.1 * zx, 1.1 * zx)
    ins = (Ls + stub + 0.50, -0.78, 1.55, 1.55)  # inset position (data units of the panel)
    y_top, y_bot = 1.47, -1.50
    kw_dim = dict(arrowstyle="<->", lw=0.6, color=INK2, shrinkA=0, shrinkB=0)
    kw_ptr = dict(arrowstyle="-|>", lw=0.5, color=INK2, shrinkA=1, shrinkB=1, mutation_scale=6)
    small = dict(fontsize=7, color=INK2)

    def draw(ax, shape, color, alpha=0.20):
        xx = Ls * np.linspace(0, 1, 1200) ** 3
        yy = radius(shape, xx, Ls, Rs)
        X, Y = np.append(xx, Ls + stub), np.append(yy, Rs)
        ax.fill_between(X, -Y, Y, color=color, alpha=alpha, lw=0)
        for s in (1, -1):
            ax.plot(X, s * Y, color=color, lw=1.1, solid_capstyle="butt")
        ax.plot([Ls, Ls], [-Rs, Rs], color=color, lw=0.5, ls=(0, (2, 2)))
        ax.plot([XL[0] + 0.12, Ls + stub + 0.15], [0, 0], color=INK2, lw=0.4, ls=(0, (8, 2, 1, 2)))
        ax.set(xlim=XL, ylim=YL, aspect="equal")
        ax.axis("off")

    def tip_inset(ax, shape, color):
        a = ax.inset_axes(ins, transform=ax.transData)
        xx = zx * np.linspace(0, 1, 800) ** 3
        yy = radius(shape, xx, Ls, Rs)
        a.fill_between(xx, -yy, yy, color=color, alpha=0.20, lw=0)
        a.plot(xx, yy, color=color, lw=1.0)
        a.plot(xx, -yy, color=color, lw=1.0)
        a.plot(win, [0, 0], color=INK2, lw=0.4, ls=(0, (8, 2, 1, 2)))
        h = 0.5 * (win[1] - win[0])
        a.set(xlim=win, ylim=(-h, h), xticks=[], yticks=[])
        a.minorticks_off()
        for sp in a.spines.values():
            sp.set(lw=0.5, color=INK2)
        a.text(0.5, 1.04, r"tip region, $\times%d$" % round(ins[2] / (win[1] - win[0])),
               transform=a.transAxes, ha="center", va="bottom", **small)
        ax.add_patch(Rectangle((win[0], -h), 2 * h, 2 * h, fill=False, lw=0.5, ec=INK2))

    def kink(ax, shape, color):
        th = np.arctan(slope(shape, Ls, Ls, Rs))
        e = 0.85
        ax.plot([Ls, Ls + e], [Rs, Rs + e * np.tan(th)], color=color, lw=0.7, ls=(0, (2, 1.5)))
        ax.add_patch(Arc((Ls, Rs), 1.3, 1.3, theta1=0, theta2=np.degrees(th), lw=0.6, color=INK))
        lab = (r"shoulder kink $\theta_{\rm sh}=\theta_c$" if shape == "cone"
               else r"shoulder kink $\theta_{\rm sh}=\arctan(nR/L)$")
        ax.annotate(lab, xy=(Ls + 0.55, Rs + 0.07), xytext=(Ls + 0.95, y_top), ha="right",
                    va="center", arrowprops=kw_ptr, fontsize=7, color=INK)

    def tangent(ax):
        ax.annotate("tangent to the body at the shoulder", xy=(Ls, Rs), xytext=(Ls + 0.95, y_top),
                    ha="right", va="center", arrowprops=kw_ptr, **small)

    def bottom(ax, *lines):
        for i, s in enumerate(lines):
            ax.text(xc, y_bot + 0.17 * (len(lines) - 1) - 0.34 * i, s, ha="center", va="center", **small)

    fig, axs = plt.subplots(4, 2, figsize=(fs.WIDTH, 1.08 * fs.WIDTH))
    axs = axs.ravel()
    titles = {
        "cone": r"Conical: $\eta=\xi$",
        "ogive": r"Tangent ogive: circular arc",
        "parabolic": r"Parabolic ($K'=1$): $\eta=2\xi-\xi^2$",
        "power050": r"Power series, $n=\tfrac12$: $\eta=\xi^{1/2}$",
        "power075": r"Power series, $n=\tfrac34$: $\eta=\xi^{3/4}$",
        "lvhaack": r"LV-Haack: Haack series, $C=\tfrac13$",
        "vonkarman": r"Von K\'arm\'an: Haack series, $C=0$",
    }

    # (a) notation, on a neutral Von Karman-like body
    ax = axs[0]
    draw(ax, "vonkarman", INK2, alpha=0.12)
    ax.set_title(r"\textbf{(a)} Notation", loc="left", fontsize=8.5, pad=2)
    ax.annotate("", xy=(0, 1.32), xytext=(Ls, 1.32), arrowprops=kw_dim)
    ax.plot([0, 0], [0.06, 1.44], color=INK2, lw=0.4)
    ax.plot([Ls, Ls], [Rs + 0.04, 1.44], color=INK2, lw=0.4)
    ax.text(Ls / 2, 1.40, r"$L$", ha="center", va="bottom", fontsize=8)
    xd = Ls + 0.55 * stub
    ax.annotate("", xy=(xd, -Rs), xytext=(xd, Rs), arrowprops=kw_dim)
    ax.text(xd + 0.10, 0.0, r"$D=2R$", ha="left", va="center", rotation=90, fontsize=8)
    xs = 0.40 * Ls
    ys = radius("vonkarman", xs, Ls, Rs)
    ax.annotate("", xy=(xs, ys), xytext=(xs, 0), arrowprops=kw_dim)
    ax.text(xs + 0.08, 0.32, r"$y(x)$", ha="left", va="center", fontsize=8)
    ax.annotate("", xy=(xs, -0.20), xytext=(0, -0.20), arrowprops=dict(kw_dim, arrowstyle="->"))
    ax.text(xs / 2, -0.27, r"$x$", ha="center", va="top", fontsize=8)
    ax.annotate("apex", xy=(0, 0.02), xytext=(-0.28, 0.78), ha="center", va="bottom",
                arrowprops=kw_ptr, **small)
    ax.annotate("shoulder", xy=(Ls, -Rs), xytext=(Ls - 0.75, -1.45), ha="center", va="center",
                arrowprops=kw_ptr, **small)
    ax.text(Ls + stub / 2, -1.45, "body tube", ha="left", va="center", **small)
    ax.annotate("", xy=(0.45, -1.05), xytext=(-0.48, -1.05), arrowprops=dict(kw_ptr, mutation_scale=7))
    ax.text(-0.48, -0.92, r"$V_\infty$", ha="left", va="bottom", fontsize=8)
    tx = Ls + stub + 0.45
    for i, s in enumerate([r"$\xi = x/L$", r"$\eta = y/R$", r"fineness $L/D$",
                           r"$\theta_{\rm sh}$: shoulder angle"]):
        ax.text(tx, 0.62 - 0.36 * i, s, ha="left", va="center", fontsize=7.5, color=INK)

    # (b)-(h) the seven profiles
    for k, shape in enumerate(ORDER):
        ax, c = axs[k + 1], COLORS[shape]
        draw(ax, shape, c)
        ax.set_title(rf"\textbf{{({'bcdefgh'[k]})}} " + titles[shape], loc="left", fontsize=8.5, pad=2)
        tip_inset(ax, shape, c)
        if shape == "cone":
            thc = np.degrees(np.arctan(Rs / Ls))
            ax.add_patch(Arc((0, 0), 2.4, 2.4, theta1=0, theta2=thc, lw=0.6, color=INK))
            ax.text(1.28, 0.06, r"$\theta_c=\arctan(R/L)$", fontsize=7, color=INK, va="bottom")
            kink(ax, shape, c)
            bottom(ax, "straight generator, sharp tip; simplest mould")
        elif shape == "ogive":
            rho = (Rs**2 + Ls**2) / (2 * Rs)
            xa = 0.45 * Ls
            ya = radius(shape, xa, Ls, Rs)
            d = np.array([Ls - xa, (Rs - rho) - ya])
            d /= np.hypot(*d)
            t = (-1.22 - ya) / d[1]
            ax.annotate("", xy=(xa + t * d[0], ya + t * d[1]), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=c, ls=(0, (3, 2)), mutation_scale=6))
            ax.annotate("", xy=(Ls, -1.22), xytext=(Ls, -Rs),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=c, ls=(0, (3, 2)), mutation_scale=6))
            ax.text(xa + 0.5 * t * d[0] - 0.10, ya + 0.5 * t * d[1], r"$\rho$", fontsize=8, ha="right")
            tangent(ax)
            bottom(ax, r"arc of radius $\rho=(R^2+L^2)/2R$,", "centre in the base plane, below the axis")
        elif shape == "parabolic":
            tangent(ax)
            bottom(ax, "parabola with its vertex at the shoulder;", "sharp tip with a finite angle")
        elif shape == "power050":
            kink(ax, shape, c)
            bottom(ax, r"blunt, rounded tip,", r"curvature radius $R^2/2L$ at the apex")
        elif shape == "power075":
            kink(ax, shape, c)
            bottom(ax, "pointed tip with a vertical tangent", "(zero tip radius)")
        elif shape == "lvhaack":
            tangent(ax)
            bottom(ax, r"least wave drag for given $L$ and volume;", "the fullest of the seven shapes")
        elif shape == "vonkarman":
            tangent(ax)
            bottom(ax, r"least wave drag for given $L$ and base area;", "pointed tip with a vertical tangent")
    fig.subplots_adjust(hspace=0.22, wspace=0.05)
    fs.save(fig, "fig_schematics", FIG)


def main():
    fs.setup()
    which = sys.argv[1:] or ["gallery", "fields", "breakdown", "robustness"]
    k = float(open(os.path.join(RES, "calibration.txt")).read().split("=")[1])
    for w in which:
        print("figure", w, flush=True)
        if w == "gallery":
            fig_gallery()
        elif w == "fields":
            fig_fields(1.4)
        elif w == "fields12":
            fig_fields(1.2)
        elif w == "breakdown":
            fig_breakdown(k)
        elif w == "grid":
            fig_grid()
        elif w == "schematics":
            fig_schematics()
        elif w == "robustness":
            fig_robustness()


if __name__ == "__main__":
    main()

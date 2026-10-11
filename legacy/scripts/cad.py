"""3D renders and watertight STL solids of every candidate nose (units: mm).

STL: surface of revolution of the profile + 60 mm shoulder, closed by a flat base disc.
Files go to results/cad/; the render goes to figures/fig_renders.pdf.
"""
import os
import struct
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import COLORS, INK  # noqa: E402
from geometry import SHAPES, surface  # noqa: E402
from figures import SHORT, FINAL, FIG, RES  # noqa: E402

R = 0.09
SH = 0.060
CAD = os.path.join(RES, "cad")


def outline(shape, f, rn=0.0, n=220):
    """Meridian (x, r) in mm from the tip, including the shoulder; x starts at 0."""
    L = 2 * R * f
    x, y = surface(shape, L, R, rn=rn, n=n)
    x = np.concatenate([x, [L + SH]])
    y = np.concatenate([y, [R]])
    return 1e3 * (x - x[0]), 1e3 * y


def write_stl(path, x, r, n_az=128, name="nose"):
    """Binary STL of the closed solid of revolution about the x axis."""
    th = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    P = np.stack([np.repeat(x[:, None], n_az, 1), r[:, None] * np.cos(th), r[:, None] * np.sin(th)], -1)
    tris = []
    for i in range(len(x) - 1):
        for j in range(n_az):
            jn = (j + 1) % n_az
            a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, jn], P[i, jn]
            if r[i] > 1e-9:          # outward normals (right-hand rule)
                tris.append((a, c, b))
                tris.append((a, d, c))
            else:                     # apex: single fan triangle
                tris.append((a, c, b))
    xe = x[-1]
    centre = np.array([xe, 0.0, 0.0])
    for j in range(n_az):          # base disc
        jn = (j + 1) % n_az
        tris.append((centre, P[-1, j], P[-1, jn]))
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            nrm = np.cross(b - a, c - a)
            nn = np.linalg.norm(nrm)
            nrm = nrm / nn if nn > 0 else nrm
            fh.write(struct.pack("<12fH", *nrm, *a, *b, *c, 0))
    return len(tris)


def render(ax, x, r, color, tip_x=None, n_az=90, zoom=1.9):
    th = np.linspace(0, 2 * np.pi, n_az)
    Xs = np.repeat(x[:, None], n_az, 1)
    Ys = r[:, None] * np.cos(th)
    Zs = r[:, None] * np.sin(th)
    ls = LightSource(azdeg=300, altdeg=35)
    base = np.array(to_rgb(color))
    rgb = np.ones(Xs.shape + (3,)) * (0.55 * base + 0.45)
    if tip_x is not None:
        rgb[Xs <= tip_x] = np.array(to_rgb("#9aa4ad"))
    shaded = ls.shade_rgb(rgb, Zs, fraction=0.9, blend_mode="soft")
    ax.plot_surface(Xs, Ys, Zs, facecolors=shaded, linewidth=0, antialiased=False, shade=False,
                    rstride=1, cstride=1)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((x.max() - x.min(), 2 * r.max(), 2 * r.max()), zoom=zoom)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-r.max(), r.max())
    ax.set_zlim(-r.max(), r.max())
    ax.view_init(elev=14, azim=-78)
    ax.set_axis_off()


def main():
    fs.setup()
    os.makedirs(CAD, exist_ok=True)
    f = FINAL["f"]
    for s in SHAPES:
        x, r = outline(s, f)
        n = write_stl(os.path.join(CAD, f"nose_{s}_LD{f:g}.stl"), x, r, name=f"{s} L/D={f}")
        print(s, n, "triangles")
    rn = FINAL["rn_over_R"] * R
    x, r = outline(FINAL["shape"], f, rn=rn, n=400)
    write_stl(os.path.join(CAD, f"nose_{FINAL['shape']}_LD{f:g}_rn{1e3 * rn:g}mm_FINAL.stl"), x, r,
              name="Weisshorn nose cone FINAL")
    # render sheet: 7 candidates + the final design (larger)
    fig = plt.figure(figsize=(fs.WIDTH, 0.80 * fs.WIDTH))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.5], hspace=0.0, wspace=0.0,
                          left=0.0, right=1.0, top=0.97, bottom=0.0)
    for k, s in enumerate(SHAPES):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection="3d")
        x, r = outline(s, f, n=160)
        render(ax, x, r, COLORS[s])
        ax.text2D(0.02, 0.80, SHORT[s], transform=ax.transAxes, fontsize=8, color=INK)
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, rf"all at $L/D = {f:g}$ ($L = {1e3 * 2 * R * f:.0f}$ mm), $D = 180$ mm," "\n"
            r"with a 60 mm shoulder; same scale", ha="center", va="center", fontsize=7.5, color=INK)
    ax = fig.add_subplot(gs[4, :], projection="3d")
    x, r = outline(FINAL["shape"], f, rn=rn, n=300)
    render(ax, x, r, COLORS[FINAL["shape"]], tip_x=80.0)
    ax.set_title(r"\textbf{Recommended:} Von K\'arm\'an, $L/D=4.5$, 4.5 mm Al 6082 tip insert (grey)",
                 fontsize=8.5, pad=-2, color=INK)
    fig.savefig(os.path.join(FIG, "fig_renders.pdf"), dpi=300, bbox_inches=None)
    fig.savefig(os.path.join(FIG, "fig_renders.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

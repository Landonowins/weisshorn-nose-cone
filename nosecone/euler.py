"""Axisymmetric Euler solver for the forebody (nose + cylinder) pressure drag.

Cell-centred finite volumes on a body-fitted C-type grid that starts on the stagnation
streamline upstream of the tip, wraps the nose and follows the cylinder. HLLC fluxes, MUSCL
(minmod) reconstruction of primitive variables, SSP-RK2 with local time stepping to reach the
steady state. Non-dimensionalisation: rho_inf = 1, c_inf = 1 (p_inf = 1/gamma), lengths
scaled by the base radius R, so every result is independent of the vehicle diameter.

Axisymmetric FV form (per radian): d(U V)/dt + sum_f F.n l_f r_f = (0, 0, p A, 0), with
V = A r_c, which preserves a uniform flow exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.optimize import brentq

from . import constants as C
from .geometry import Profile, surface

GAMMA = C.GAMMA


# --------------------------------------------------------------------------- grid
@dataclass
class GridSpec:
    """Grid resolution and domain size (lengths in units of the base radius R)."""

    ni_nose: int = 120          # cells along the nose
    ni_cyl: int = 40            # cells along the cylinder
    nj: int = 80                # cells normal to the wall
    wall_dy: float = 2e-3       # first-cell height / R
    l_cyl: float = 5.0          # cylinder length behind the nose / R
    d_up: float = 1.0           # outer boundary ahead of the tip / R (times 1 + 0.3 L/D)
    out_angle: float = 72.0     # asymptotic angle of the outer boundary [deg]
    smooth_iters: int = 0       # optional Laplacian smoothing passes (0 = off)

    @classmethod
    def from_level(cls, level: list, l_cyl: float = 5.0, d_up: float = 1.0, out_angle: float = 72.0) -> "GridSpec":
        """Grid from a level [ni_nose, ni_cyl, nj, wall_dy] of the configuration."""
        ni, nc, nj, dy = level
        return cls(int(ni), int(nc), int(nj), float(dy), l_cyl, d_up, out_angle)

    @property
    def n_cells(self) -> int:
        """Number of cells of the grid."""
        return (self.ni_nose + self.ni_cyl) * self.nj


def _stretch(n: int, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """n+1 points in [0, 1] clustered at 0 (strength a) and/or 1 (strength b)."""
    s = np.linspace(0.0, 1.0, n + 1)
    if a > 0 and b > 0:
        return 0.5 * (1 + np.tanh(a * (s - 0.5)) / np.tanh(0.5 * a)) if a == b else _two_sided(s, a, b)
    if a > 0:
        return 1 - np.tanh(a * (1 - s)) / np.tanh(a)
    if b > 0:
        return np.tanh(b * s) / np.tanh(b)
    return s


def _two_sided(s: np.ndarray, a: float, b: float) -> np.ndarray:
    """Blend of one-sided stretchings, monotone for a, b > 0."""
    fa = 1 - np.tanh(a * (1 - s)) / np.tanh(a)
    fb = np.tanh(b * s) / np.tanh(b)
    w = s
    return (1 - w) * fa + w * fb


def _resample(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points at normalised arc length sigma along the polyline (x, y)."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    d /= d[-1]
    return np.interp(sigma, d, x), np.interp(sigma, d, y)


def _geom(n: int, d0: float) -> np.ndarray:
    """n+1 points in [0, 1] with first spacing d0 and constant growth ratio."""
    if d0 * n >= 1:
        return np.linspace(0, 1, n + 1)
    g = brentq(lambda g: d0 * (g**n - 1) / (g - 1) - 1, 1.0 + 1e-9, 2.0)
    d = d0 * g ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(d)]) / np.sum(d)


def _smooth(X: np.ndarray, Y: np.ndarray, iters: int = 40, w: float = C.GRID_SMOOTH_WEIGHT,
            j0: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Light Laplacian smoothing along i of the nodes away from the wall (j >= j0)."""
    X, Y = X.copy(), Y.copy()
    for _ in range(iters):
        Xs = X.copy()
        Ys = Y.copy()
        Xs[1:-1, j0:-1] = (1 - w) * X[1:-1, j0:-1] + w * 0.5 * (X[2:, j0:-1] + X[:-2, j0:-1])
        Ys[1:-1, j0:-1] = (1 - w) * Y[1:-1, j0:-1] + w * 0.5 * (Y[2:, j0:-1] + Y[:-2, j0:-1])
        X, Y = Xs, Ys
    return X, Y


def make_grid(profile: Union[Profile, str], fineness: float, rn: float = 0.0,
              spec: Optional[GridSpec] = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Body-fitted grid around a nose of fineness L/D followed by a cylinder (R = 1).

    Args:
        profile: nose profile.
        fineness: L/D.
        rn: spherical tip radius / R (0 = sharp).
        spec: grid resolution and domain (default: the medium level).

    Returns:
        Node arrays X, Y of shape (ni+1, nj+1) and the index of the shoulder.
    """
    g = spec or GridSpec()
    R = 1.0
    L = 2.0 * fineness * R
    xb, yb = surface(profile, L, R, rn, n=C.GRID_BODY_SAMPLES)
    x0 = xb[0]                                   # actual tip (a blunted nose starts later)
    # body points: nose clustered at tip and shoulder, then cylinder
    sig = _two_sided(np.linspace(0, 1, g.ni_nose + 1), C.GRID_TIP_CLUSTER, C.GRID_SHOULDER_CLUSTER)
    xn, yn = _resample(xb, yb, sig)
    xc = L + g.l_cyl * (_stretch(g.ni_cyl, a=C.GRID_CYL_CLUSTER)[1:])
    Xb = np.concatenate([xn, xc])
    Yb = np.concatenate([yn, np.full(g.ni_cyl, R)])
    # outer boundary: hyperbola, vertical where it crosses the axis
    xa = x0 - g.d_up * (1 + C.GRID_UPSTREAM_SLOPE * fineness)
    k = 1.0 / np.tan(np.radians(g.out_angle))
    b = C.GRID_HYPERBOLA_B
    r_end = (Xb[-1] - xa) / k + C.GRID_OUTER_MARGIN
    rr = np.linspace(0.0, r_end, C.GRID_OUTER_SAMPLES)
    xo, yo = xa + k * (np.sqrt(rr**2 + b**2) - b), rr
    keep = xo <= Xb[-1]
    xo, yo = xo[keep], yo[keep]
    xo = np.append(xo, Xb[-1])
    yo = np.append(yo, yo[-1] + (Xb[-1] - xo[-2]) / k)
    # body arc-length parameter mapped to the outer boundary (less clustered)
    db = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(Xb), np.diff(Yb)))])
    sb = db / db[-1]
    so = sb**C.GRID_OUTER_EXPONENT
    Xo, Yo = _resample(xo, yo, so)
    # body normals (the first line runs up the stagnation streamline, along -x)
    tx = np.gradient(Xb)
    ty = np.gradient(Yb)
    tn = np.hypot(tx, ty)
    nxb, nyb = -ty / tn, tx / tn
    nxb[0], nyb[0] = -1.0, 0.0
    # radial distribution: geometric growth from the wall; lines start along the body normal
    # (orthogonal near-wall cells) and bend towards the outer point
    eta = _geom(g.nj, g.wall_dy)
    dx, dy = Xo - Xb, Yo - Yb
    D = np.hypot(dx, dy)
    w = (np.exp(-eta / C.GRID_NORMAL_BLEND) * (1 - eta))[None, :]
    ex = w * nxb[:, None] + (1 - w) * (dx / D)[:, None]
    ey = w * nyb[:, None] + (1 - w) * (dy / D)[:, None]
    X = Xb[:, None] + (eta[None, :] * D[:, None]) * ex
    Y = Yb[:, None] + (eta[None, :] * D[:, None]) * ey
    X[:, -1], Y[:, -1] = Xo, Yo
    Y[0, :] = 0.0
    X, Y = _smooth(X, Y, iters=g.smooth_iters, j0=g.nj // 3) if g.smooth_iters else (X, Y)
    return X, Y, g.ni_nose


# ----------------------------------------------------------------- geometry metrics
class Mesh:
    """Face lengths, normals, radii, cell areas and centroids of a structured grid."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, i_shoulder: int):
        self.X, self.Y = X, Y
        self.ni, self.nj = X.shape[0] - 1, X.shape[1] - 1
        self.i_sh = i_shoulder
        # i-faces: edge (i,j)->(i,j+1), normal towards +i
        tx, ty = X[:, 1:] - X[:, :-1], Y[:, 1:] - Y[:, :-1]
        self.li = np.hypot(tx, ty)
        self.nix, self.niy = ty / self.li, -tx / self.li
        self.ri = 0.5 * (Y[:, 1:] + Y[:, :-1])
        # j-faces: edge (i,j)->(i+1,j), normal towards +j
        tx, ty = X[1:, :] - X[:-1, :], Y[1:, :] - Y[:-1, :]
        self.lj = np.hypot(tx, ty)
        self.njx, self.njy = -ty / self.lj, tx / self.lj
        self.rj = 0.5 * (Y[1:, :] + Y[:-1, :])
        # cells (counter-clockwise quad)
        xa, ya = X[:-1, :-1], Y[:-1, :-1]
        xb, yb = X[1:, :-1], Y[1:, :-1]
        xc, yc = X[1:, 1:], Y[1:, 1:]
        xd, yd = X[:-1, 1:], Y[:-1, 1:]
        cr = [(xa, ya, xb, yb), (xb, yb, xc, yc), (xc, yc, xd, yd), (xd, yd, xa, ya)]
        A = sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in cr) / 2
        cy = sum((y1 + y2) * (x1 * y2 - x2 * y1) for x1, y1, x2, y2 in cr) / (6 * A)
        cx = sum((x1 + x2) * (x1 * y2 - x2 * y1) for x1, y1, x2, y2 in cr) / (6 * A)
        if A.min() <= 0:
            raise ValueError(f"negative cell area: min {A.min():.3e}")
        self.A, self.xc, self.yc = A, cx, cy
        self.V = A * cy

    def wall_drag(self, cp: np.ndarray) -> float:
        """C_D,w from a wall C_p distribution: -2 sum cp n_x l r over the nose faces (R = 1)."""
        sl = slice(0, self.i_sh)
        return -2 * np.sum(cp[sl] * self.njx[sl, 0] * self.lj[sl, 0] * self.rj[sl, 0])


# ------------------------------------------------------------------- flux functions
def prim2cons(W: np.ndarray) -> np.ndarray:
    """Primitive (rho, u, v, p) to conservative (rho, rho u, rho v, E) variables."""
    rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    E = p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=-1)


def cons2prim(U: np.ndarray) -> np.ndarray:
    """Conservative to primitive variables."""
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    p = (GAMMA - 1) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
    return np.stack([rho, u, v, p], axis=-1)


def hllc(WL: np.ndarray, WR: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HLLC flux through a face with unit normal (nx, ny); returns (F, p_star)."""
    rL, uL, vL, pL = (WL[..., k] for k in range(4))
    rR, uR, vR, pR = (WR[..., k] for k in range(4))
    unL, unR = uL * nx + vL * ny, uR * nx + vR * ny
    utL, utR = -uL * ny + vL * nx, -uR * ny + vR * nx
    cL, cR = np.sqrt(GAMMA * pL / rL), np.sqrt(GAMMA * pR / rR)
    EL = pL / (GAMMA - 1) + 0.5 * rL * (unL**2 + utL**2)
    ER = pR / (GAMMA - 1) + 0.5 * rR * (unR**2 + utR**2)
    SL = np.minimum(unL - cL, unR - cR)
    SR = np.maximum(unL + cL, unR + cR)
    Ss = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / (rL * (SL - unL) - rR * (SR - unR))

    def flux(r, un, ut, p, E):
        return np.stack([r * un, r * un * un + p, r * un * ut, (E + p) * un], axis=-1)

    def ustate(r, un, ut, E):
        return np.stack([r, r * un, r * ut, E], axis=-1)

    def ustar(r, un, ut, p, E, S):
        f = r * (S - un) / (S - Ss)
        return np.stack([f, f * Ss, f * ut, f * (E / r + (Ss - un) * (Ss + p / (r * (S - un))))], axis=-1)

    FL, FR = flux(rL, unL, utL, pL, EL), flux(rR, unR, utR, pR, ER)
    FsL = FL + SL[..., None] * (ustar(rL, unL, utL, pL, EL, SL) - ustate(rL, unL, utL, EL))
    FsR = FR + SR[..., None] * (ustar(rR, unR, utR, pR, ER, SR) - ustate(rR, unR, utR, ER))
    F = np.where((SL >= 0)[..., None], FL,
                 np.where((Ss >= 0)[..., None], FsL, np.where((SR > 0)[..., None], FsR, FR)))
    pstar = pL + rL * (SL - unL) * (Ss - unL)
    # rotate momentum back to (x, r)
    Fx = F[..., 1] * nx - F[..., 2] * ny
    Fy = F[..., 1] * ny + F[..., 2] * nx
    return np.stack([F[..., 0], Fx, Fy, F[..., 3]], axis=-1), pstar


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod slope limiter."""
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


# ----------------------------------------------------------------------- the solver
class EulerAxi:
    """Steady axisymmetric Euler solver (march in pseudo-time to convergence)."""

    def __init__(self, mesh: Mesh, mach: float, cfl: float = 0.6, order: int = 2):
        self.m = mesh
        self.M = mach
        self.cfl = cfl
        self.order = order
        self.Winf = np.array([1.0, mach, 0.0, 1.0 / GAMMA])
        W = np.broadcast_to(self.Winf, (mesh.ni, mesh.nj, 4)).copy()
        self.U = prim2cons(W)
        self.history: list[tuple[int, float, float]] = []

    def set_mach(self, mach: float) -> None:
        """Change the free-stream Mach number, keeping the current flow as initial guess."""
        self.M = mach
        self.Winf = np.array([1.0, mach, 0.0, 1.0 / GAMMA])
        self.history = []

    # ghost cells ---------------------------------------------------------------
    def _farfield(self, Wi: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
        """Characteristic (Riemann-invariant) far-field state for outward normal n."""
        rho, u, v, p = (Wi[..., k] for k in range(4))
        c = np.sqrt(GAMMA * p / rho)
        ri, ui, vi, pi_ = self.Winf
        ci = np.sqrt(GAMMA * pi_ / ri)
        vn, vni = u * nx + v * ny, ui * nx + vi * ny
        Rp = vn + 2 * c / (GAMMA - 1)
        Rm = vni - 2 * ci / (GAMMA - 1)
        vnb = 0.5 * (Rp + Rm)
        cb = 0.25 * (GAMMA - 1) * (Rp - Rm)
        out = vnb > 0
        s = np.where(out, p / rho**GAMMA, pi_ / ri**GAMMA)
        ut = np.where(out, -u * ny + v * nx, -ui * ny + vi * nx)
        rb = (cb**2 / (GAMMA * s)) ** (1 / (GAMMA - 1))
        pb = rb * cb**2 / GAMMA
        ub, vb = vnb * nx - ut * ny, vnb * ny + ut * nx
        Wb = np.stack([rb, ub, vb, pb], axis=-1)
        sup_in = (vn <= -c)
        sup_out = (vn >= c)
        Wb = np.where(sup_in[..., None], np.broadcast_to(self.Winf, Wb.shape), Wb)
        Wb = np.where(sup_out[..., None], Wi, Wb)
        return Wb

    def _padded(self, W: np.ndarray) -> np.ndarray:
        m = self.m
        ni, nj = m.ni, m.nj
        P = np.empty((ni + 4, nj + 4, 4))
        P[2:-2, 2:-2] = W
        # wall (j = 0): mirror normal velocity about the wall normal
        nx, ny = m.njx[:, 0], m.njy[:, 0]
        for g, src in ((1, 0), (0, 1)):
            Ws = W[:, src].copy()
            vn = Ws[:, 1] * nx + Ws[:, 2] * ny
            Ws[:, 1] -= 2 * vn * nx
            Ws[:, 2] -= 2 * vn * ny
            P[2:-2, g] = Ws
        # far field (j = nj)
        Wb = self._farfield(W[:, -1], m.njx[:, -1], m.njy[:, -1])
        P[2:-2, -2] = Wb
        P[2:-2, -1] = Wb
        # axis (i = 0): mirror radial velocity
        for g, src in ((1, 0), (0, 1)):
            P[g, 2:-2] = P[2 + src, 2:-2] * np.array([1, 1, -1, 1])
        # outflow (i = ni)
        Wb = self._farfield(W[-1, :], m.nix[-1, :], m.niy[-1, :])
        P[-2, 2:-2] = Wb
        P[-1, 2:-2] = Wb
        # corners (not used by the dimension-by-dimension stencil, filled for safety)
        P[:2, :2] = P[2:4, 2:4]
        P[:2, -2:] = P[2:4, -4:-2]
        P[-2:, :2] = P[-4:-2, 2:4]
        P[-2:, -2:] = P[-4:-2, -4:-2]
        return P

    # residual --------------------------------------------------------------------
    def _faces(self, P: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Left/right states on all faces along `axis` (0: i-faces, 1: j-faces)."""
        if axis == 0:
            Wm2, Wm1, W0, Wp1 = P[:-3, 2:-2], P[1:-2, 2:-2], P[2:-1, 2:-2], P[3:, 2:-2]
        else:
            Wm2, Wm1, W0, Wp1 = P[2:-2, :-3], P[2:-2, 1:-2], P[2:-2, 2:-1], P[2:-2, 3:]
        if self.order == 1:
            return Wm1, W0
        d0 = W0 - Wm1
        WL = Wm1 + 0.5 * minmod(Wm1 - Wm2, d0)
        WR = W0 - 0.5 * minmod(d0, Wp1 - W0)
        bad = (WL[..., 0] <= 0) | (WL[..., 3] <= 0) | (WR[..., 0] <= 0) | (WR[..., 3] <= 0)
        if bad.any():
            WL = np.where(bad[..., None], Wm1, WL)
            WR = np.where(bad[..., None], W0, WR)
        return WL, WR

    def residual(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Net flux out of every cell minus the axisymmetric source; also returns W."""
        m = self.m
        W = cons2prim(U)
        P = self._padded(W)
        WL, WR = self._faces(P, 0)
        Fi, _ = hllc(WL, WR, m.nix, m.niy)
        Fi *= (m.li * m.ri)[..., None]
        WL, WR = self._faces(P, 1)
        Fj, pst = hllc(WL, WR, m.njx, m.njy)
        self.p_wall = pst[:, 0]
        Fj *= (m.lj * m.rj)[..., None]
        Res = Fi[1:] - Fi[:-1] + Fj[:, 1:] - Fj[:, :-1]
        Res[..., 2] -= W[..., 3] * m.A
        return Res, W

    def local_dt(self, W: np.ndarray) -> np.ndarray:
        """Local (planar) time step of every cell from the CFL condition."""
        m = self.m
        c = np.sqrt(GAMMA * W[..., 3] / W[..., 0])
        sx = 0.5 * (m.nix[1:] * m.li[1:] + m.nix[:-1] * m.li[:-1])
        sy = 0.5 * (m.niy[1:] * m.li[1:] + m.niy[:-1] * m.li[:-1])
        tx = 0.5 * (m.njx[:, 1:] * m.lj[:, 1:] + m.njx[:, :-1] * m.lj[:, :-1])
        ty = 0.5 * (m.njy[:, 1:] * m.lj[:, 1:] + m.njy[:, :-1] * m.lj[:, :-1])
        lam = (np.abs(W[..., 1] * sx + W[..., 2] * sy) + c * np.hypot(sx, sy)
               + np.abs(W[..., 1] * tx + W[..., 2] * ty) + c * np.hypot(tx, ty))
        return self.cfl * m.A / lam

    # forces ----------------------------------------------------------------------
    def nose_drag(self, p_wall: Optional[np.ndarray] = None) -> float:
        """Forebody pressure-drag coefficient (reference area pi R^2, R = 1)."""
        m = self.m
        p = self.p_wall if p_wall is None else p_wall
        sl = slice(0, m.i_sh)
        q = 0.5 * self.M**2  # rho_inf = 1
        Fx = -np.sum((p[sl] - 1 / GAMMA) * m.njx[sl, 0] * m.lj[sl, 0] * m.rj[sl, 0]) * 2 * np.pi
        return Fx / (q * np.pi)

    def wall_cp(self) -> tuple[np.ndarray, np.ndarray]:
        """Wall pressure coefficient at the face centres (x / R, C_p)."""
        q = 0.5 * self.M**2
        xw = 0.5 * (self.m.X[1:, 0] + self.m.X[:-1, 0])
        return xw, (self.p_wall - 1 / GAMMA) / q

    # driver ----------------------------------------------------------------------
    def run(self, max_iter: int = 20000, tol: float = 2e-5, check: int = 200,
            first_order_iters: int = 300, verbose: bool = False) -> float:
        """March to the steady state; stop when C_D,w changes by < tol (relative) over `check`."""
        order = self.order
        cd_old = None
        for it in range(1, max_iter + 1):
            self.order = 1 if it <= first_order_iters else order
            R1, W = self.residual(self.U)
            # planar CFL time step; the axisymmetric update divides by V = A r_c
            dtV = (self.local_dt(W) / self.m.V)[..., None]
            U1 = self.U - dtV * R1
            R2, _ = self.residual(U1)
            self.U = 0.5 * (self.U + U1 - dtV * R2)
            if it % check == 0:
                if not np.all(np.isfinite(self.U)):
                    raise FloatingPointError("solution diverged")
                cd = self.nose_drag()
                res = np.sqrt(np.mean((R1[..., 0] / self.m.V) ** 2))
                self.history.append((it, cd, res))
                if verbose:
                    print(f"{it:6d}  cd={cd:.5f}  res={res:.3e}")
                if it > first_order_iters + 4 * check and cd_old is not None:
                    if abs(cd - cd_old) < tol * max(abs(cd), 1e-3):
                        break
                cd_old = cd
        self.order = order
        self.residual(self.U)
        return self.nose_drag()

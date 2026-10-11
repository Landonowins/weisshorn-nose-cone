"""Axisymmetric Euler solver for the forebody (nose + cylinder) pressure drag.

Cell-centred finite volumes on a body-fitted C-type grid that starts on the
stagnation streamline upstream of the tip, wraps the nose and follows the cylinder.
HLLC fluxes, MUSCL (minmod) reconstruction of primitive variables, SSP-RK2 with local
time stepping to reach the steady state. Non-dimensionalisation: rho_inf = 1,
c_inf = 1 (p_inf = 1/gamma), lengths scaled by the base radius R.

Axisymmetric FV form (per radian):  d(U V)/dt + sum_f F.n l_f r_f = (0, 0, p A, 0)
with V = A r_c, which preserves a uniform flow exactly.
"""
import numpy as np

from geometry import surface

GAMMA = 1.4


# --------------------------------------------------------------------------- grid
def _stretch(n, a=0.0, b=0.0):
    """n+1 points in [0,1] clustered at 0 (strength a) and/or 1 (strength b)."""
    s = np.linspace(0.0, 1.0, n + 1)
    if a > 0 and b > 0:
        return 0.5 * (1 + np.tanh(a * (s - 0.5)) / np.tanh(0.5 * a)) if a == b else _two_sided(s, a, b)
    if a > 0:
        return 1 - np.tanh(a * (1 - s)) / np.tanh(a)
    if b > 0:
        return np.tanh(b * s) / np.tanh(b)
    return s


def _two_sided(s, a, b):
    # blend of one-sided stretchings, monotone for a, b > 0
    fa = 1 - np.tanh(a * (1 - s)) / np.tanh(a)
    fb = np.tanh(b * s) / np.tanh(b)
    w = s
    return (1 - w) * fa + w * fb


def _resample(x, y, sigma):
    """Points at normalised arc length sigma along the polyline (x, y)."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    d /= d[-1]
    return np.interp(sigma, d, x), np.interp(sigma, d, y)


def make_grid(shape, fineness, rn=0.0, ni_nose=120, ni_cyl=40, nj=80, l_cyl=5.0,
              d_up=1.0, out_angle=72.0, wall_dy=2e-3, smooth_iters=0):
    """Body-fitted grid; lengths in units of the base radius R (R = 1).

    Returns node arrays X, Y of shape (ni+1, nj+1) and the index of the shoulder.
    """
    R = 1.0
    L = 2.0 * fineness * R
    xb, yb = surface(shape, L, R, rn, n=3000)
    x0 = xb[0]                                   # actual tip (blunted nose starts later)
    # body points: nose clustered at tip and shoulder, then cylinder
    sig = _two_sided(np.linspace(0, 1, ni_nose + 1), 4.0, 2.0)
    xn, yn = _resample(xb, yb, sig)
    xc = L + l_cyl * (_stretch(ni_cyl, a=2.5)[1:])
    Xb = np.concatenate([xn, xc])
    Yb = np.concatenate([yn, np.full(ni_cyl, R)])
    # outer boundary: hyperbola, vertical where it crosses the axis
    xa = x0 - d_up * (1 + 0.3 * fineness)
    k = 1.0 / np.tan(np.radians(out_angle))
    b = 2.0
    r_end = (Xb[-1] - xa) / k + 2.0
    rr = np.linspace(0.0, r_end, 4000)
    xo, yo = xa + k * (np.sqrt(rr**2 + b**2) - b), rr
    keep = xo <= Xb[-1]
    xo, yo = xo[keep], yo[keep]
    xo = np.append(xo, Xb[-1])
    yo = np.append(yo, yo[-1] + (Xb[-1] - xo[-2]) / k)
    # body arc-length parameter mapped to the outer boundary (less clustered)
    db = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(Xb), np.diff(Yb)))])
    sb = db / db[-1]
    so = 0.5 * (sb + sb**0.5) * 0.0 + sb**0.6
    Xo, Yo = _resample(xo, yo, so)
    # body normals (the first line runs up the stagnation streamline, along -x)
    tx = np.gradient(Xb)
    ty = np.gradient(Yb)
    tn = np.hypot(tx, ty)
    nxb, nyb = -ty / tn, tx / tn
    nxb[0], nyb[0] = -1.0, 0.0
    # radial distribution: geometric growth from the wall; lines start along the
    # body normal (orthogonal near-wall cells) and bend towards the outer point
    eta = _geom(nj, wall_dy)
    dx, dy = Xo - Xb, Yo - Yb
    D = np.hypot(dx, dy)
    w = (np.exp(-eta / 0.08) * (1 - eta))[None, :]
    ex = w * nxb[:, None] + (1 - w) * (dx / D)[:, None]
    ey = w * nyb[:, None] + (1 - w) * (dy / D)[:, None]
    X = Xb[:, None] + (eta[None, :] * D[:, None]) * ex
    Y = Yb[:, None] + (eta[None, :] * D[:, None]) * ey
    X[:, -1], Y[:, -1] = Xo, Yo
    Y[0, :] = 0.0
    X, Y = _smooth(X, Y, iters=smooth_iters, j0=nj // 3) if smooth_iters else (X, Y)
    return X, Y, ni_nose


def _geom(n, d0):
    """n+1 points in [0, 1] with first spacing d0 and constant growth ratio."""
    from scipy.optimize import brentq
    if d0 * n >= 1:
        return np.linspace(0, 1, n + 1)
    g = brentq(lambda g: d0 * (g**n - 1) / (g - 1) - 1, 1.0 + 1e-9, 2.0)
    d = d0 * g ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(d)]) / np.sum(d)


def _smooth(X, Y, iters=40, w=0.3, j0=20):
    """Light Laplacian smoothing along i of the nodes away from the wall (j >= j0)."""
    X, Y = X.copy(), Y.copy()
    for _ in range(iters):
        Xs = X.copy()
        Ys = Y.copy()
        Xs[1:-1, j0:-1] = (1 - w) * X[1:-1, j0:-1] + w * 0.5 * (X[2:, j0:-1] + X[:-2, j0:-1])
        Ys[1:-1, j0:-1] = (1 - w) * Y[1:-1, j0:-1] + w * 0.5 * (Y[2:, j0:-1] + Y[:-2, j0:-1])
        X, Y = Xs, Ys
    return X, Y


# ----------------------------------------------------------------- geometry metrics
class Mesh:
    def __init__(self, X, Y, i_shoulder):
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


# ------------------------------------------------------------------- flux functions
def prim2cons(W):
    rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    E = p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=-1)


def cons2prim(U):
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    p = (GAMMA - 1) * (U[..., 3] - 0.5 * rho * (u * u + v * v))
    return np.stack([rho, u, v, p], axis=-1)


def hllc(WL, WR, nx, ny):
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


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


# ----------------------------------------------------------------------- the solver
class EulerAxi:
    def __init__(self, mesh, mach, cfl=0.6, order=2):
        self.m = mesh
        self.M = mach
        self.cfl = cfl
        self.order = order
        self.Winf = np.array([1.0, mach, 0.0, 1.0 / GAMMA])
        W = np.broadcast_to(self.Winf, (mesh.ni, mesh.nj, 4)).copy()
        self.U = prim2cons(W)
        self.history = []

    def set_mach(self, mach):
        """Change the free-stream Mach number, keeping the current flow as initial guess."""
        self.M = mach
        self.Winf = np.array([1.0, mach, 0.0, 1.0 / GAMMA])
        self.history = []

    # ghost cells ---------------------------------------------------------------
    def _farfield(self, Wi, nx, ny):
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

    def _padded(self, W):
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
    def _faces(self, P, axis):
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

    def residual(self, U):
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

    def local_dt(self, W):
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
    def nose_drag(self, p_wall=None):
        """Forebody pressure-drag coefficient (reference area pi R^2, R = 1)."""
        m = self.m
        p = self.p_wall if p_wall is None else p_wall
        sl = slice(0, m.i_sh)
        # body outward normal is +j normal; axial force from (p - p_inf)
        q = 0.5 * self.M**2  # rho_inf = 1
        Fx = -np.sum((p[sl] - 1 / GAMMA) * m.njx[sl, 0] * m.lj[sl, 0] * m.rj[sl, 0]) * 2 * np.pi
        return Fx / (q * np.pi)

    def wall_cp(self):
        q = 0.5 * self.M**2
        xw = 0.5 * (self.m.X[1:, 0] + self.m.X[:-1, 0])
        return xw, (self.p_wall - 1 / GAMMA) / q

    # driver ----------------------------------------------------------------------
    def run(self, max_iter=20000, tol=2e-5, check=200, first_order_iters=300, verbose=False):
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

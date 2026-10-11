"""Geometry export for CAD and meshing: profile CSV, watertight STL and STEP solids (units: mm).

All solids are surfaces of revolution about the x axis, with x = 0 at the (blunted) tip, made of
the curved profile, the cylindrical shoulder and a flat base disc.
"""
from __future__ import annotations

import contextlib
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .geometry import Profile, surface


def profile_points(profile: Profile, L: float, R: float, rn: float = 0.0, n: int = 200,
                   cluster: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """n profile points (x from the tip, r) [m], clustered towards the tip along the arc length."""
    x, y = surface(profile, L, R, rn=rn, n=4000)
    x0 = x[0]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sq = s[-1] * np.linspace(0, 1, n) ** cluster
    xs, ys = np.interp(sq, s, x), np.interp(sq, s, y)
    return xs - x0, ys


def write_profile_csv(path: Union[str, Path], profile: Profile, L: float, R: float, rn: float = 0.0, n: int = 200,
                      cluster: float = 1.6) -> float:
    """CSV of the outer mould line (columns x_mm, r_mm, x from the tip). Returns the physical length [mm]."""
    x, y = surface(profile, L, R, rn=rn, n=4000)
    x0 = x[0]
    # resample with points clustered at the tip
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sq = s[-1] * np.linspace(0, 1, n) ** cluster
    xs, ys = np.interp(sq, s, x), np.interp(sq, s, y)
    pd.DataFrame({"x_mm": 1e3 * (xs - x0), "r_mm": 1e3 * ys}).round(3).to_csv(path, index=False)
    return 1e3 * (L - x0)


def outline(profile: Profile, L: float, R: float, shoulder: float, rn: float = 0.0,
            n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Meridian (x, r) in mm from the tip, including the shoulder; x starts at 0."""
    x, y = surface(profile, L, R, rn=rn, n=n)
    x = np.concatenate([x, [L + shoulder]])
    y = np.concatenate([y, [R]])
    return 1e3 * (x - x[0]), 1e3 * y


def write_stl(path: Union[str, Path], x: np.ndarray, r: np.ndarray, n_az: int = 128, name: str = "nose") -> int:
    """Binary STL of the closed solid of revolution about the x axis. Returns the triangle count."""
    th = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    # an apex radius of ~1e-16 from round-off (tangent ogive) would leave a pin-hole: snap it to 0
    r = np.where(np.abs(r) <= 1e-9, 0.0, r)
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


def stl_check(path: Union[str, Path]) -> dict:
    """Closure (non-manifold edge count) and enclosed volume [mm^3] of a binary STL."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    v32 = arr["v"].reshape(-1, 3, 3) + np.float32(0.0)   # -0.0 -> +0.0 (same point)
    v = v32.astype(float)
    vol = np.sum(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))) / 6
    edges: dict = {}
    for tri in v32:                       # exact (float32) vertex keys
        k = [t.tobytes() for t in tri]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if k[a] != k[b]:
                e = (k[a], k[b]) if k[a] < k[b] else (k[b], k[a])
                edges[e] = edges.get(e, 0) + 1
    return dict(triangles=n, volume_mm3=vol, open_edges=sum(1 for c in edges.values() if c != 2))


def step_available() -> bool:
    """True if the optional gmsh package (OpenCASCADE kernel) is installed."""
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


def write_step(path: Union[str, Path], x_mm: np.ndarray, r_mm: np.ndarray, shoulder_mm: float,
               name: str = "nose") -> Optional[dict]:
    """STEP solid of revolution: interpolating spline through the profile, shoulder, base disc.

    Uses gmsh/OpenCASCADE; returns None (and writes nothing) if gmsh is not installed. The file is
    read back and its volume returned for checking.
    """
    try:
        import gmsh
    except ImportError:
        return None
    with _preserve_environment(), _quiet_stdout():
        return _write_step(gmsh, path, x_mm, r_mm, shoulder_mm, name)


def _write_step(gmsh, path, x_mm, r_mm, shoulder_mm, name):
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        occ = gmsh.model.occ
        # profile from the tip (on the axis) to the shoulder; drop duplicate points
        keep = np.concatenate([[True], np.hypot(np.diff(x_mm), np.diff(r_mm)) > 1e-9])
        xs, rs = x_mm[keep], r_mm[keep]
        rs[0] = 0.0
        pts = [occ.addPoint(float(a), float(b), 0.0) for a, b in zip(xs, rs)]
        x_end = float(xs[-1] + shoulder_mm)
        p_sh = occ.addPoint(x_end, float(rs[-1]), 0.0)
        p_base = occ.addPoint(x_end, 0.0, 0.0)
        c_prof = occ.addSpline(pts)
        c_sh = occ.addLine(pts[-1], p_sh)
        c_base = occ.addLine(p_sh, p_base)
        c_axis = occ.addLine(p_base, pts[0])
        loop = occ.addCurveLoop([c_prof, c_sh, c_base, c_axis])
        face = occ.addPlaneSurface([loop])
        occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, 2 * np.pi)
        occ.remove([(2, face)], recursive=False)
        occ.synchronize()
        gmsh.write(str(path))
        gmsh.model.remove()
        gmsh.model.add("check")
        tags = gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        vols = [t for t in tags if t[0] == 3]
        vol = sum(gmsh.model.occ.getMass(3, t[1]) for t in vols)
        return dict(volume_mm3=vol, solids=len(vols))
    finally:
        gmsh.finalize()


@contextlib.contextmanager
def _quiet_stdout():
    """Silence C-level stdout (OpenCASCADE prints transfer statistics) during a block."""
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    with open(os.devnull, "w") as dn:
        os.dup2(dn.fileno(), fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)


@contextlib.contextmanager
def _preserve_environment():
    """Restore the process environment after gmsh (gmsh.initialize truncates PATH on Windows)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        for k, v in saved.items():
            if os.environ.get(k) != v:
                os.environ[k] = v
            if sys.platform == "win32":          # also the Win32 block inherited by child processes
                import ctypes
                ctypes.windll.kernel32.SetEnvironmentVariableW(k, v)

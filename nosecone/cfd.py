"""Running Euler studies and maintaining the wave-drag database.

A *study* is a set of geometries (profile x fineness x tip radius) solved on one grid level for a
sequence of Mach numbers, from the highest down: each Mach number starts from the converged
flow of the previous one (Mach continuation). Every converged point writes the wall pressure to
``<cp_dir>/<study>_<profile>_f<L/D>_rn<r_n/R>_<grid>_M<Mach>.npz``; the database CSV is rebuilt
from these files, so interrupted runs keep every finished point.
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
import re
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .config import Config
from .euler import EulerAxi, GridSpec, Mesh, cons2prim, make_grid
from .geometry import Profile

LOG_FIELDS = ["study", "shape", "fineness", "rn", "grid", "mach", "cd", "iters", "converged", "seconds"]
TAG_RE = re.compile(r"(\w+?)_(\w+?)_f([\d.]+)_rn([\d.]+)_(\w+)_M([\d.]+)\.npz$")


@dataclass
class SolverSettings:
    """Numerical settings shared by all jobs (from the ``cfd`` section)."""

    cfl: float = 0.85
    fallback_cfl: float = 0.5
    max_iter: int = 6000
    fine_iter_factor: int = 2
    tolerance: float = 2e-5
    check: int = 250
    first_order_iters: int = 300
    grids: dict = field(default_factory=dict)
    l_cyl: float = 5.0
    d_up: float = 1.0
    out_angle: float = 72.0
    save_field_machs: tuple = (1.4, 1.2)

    @classmethod
    def from_config(cls, cfg: Config) -> "SolverSettings":
        """Settings of the ``cfd`` section of a configuration."""
        c = cfg.cfd
        return cls(c.cfl, c.fallback_cfl, c.max_iter, c.fine_iter_factor, c.tolerance, c.check,
                   c.first_order_iters, dict(c.grids), c.cylinder_length, c.upstream_distance,
                   c.outer_angle, tuple(c.save_fields))

    def grid_spec(self, level: str) -> GridSpec:
        """Grid of a level name (coarse, medium, fine, ...)."""
        return GridSpec.from_level(self.grids[level], self.l_cyl, self.d_up, self.out_angle)


@dataclass
class Job:
    """One geometry solved for a Mach sequence."""

    study: str
    profile: Profile
    fineness: float
    rn: float
    grid: str
    machs: list
    save_fields: bool = False
    max_iter: Optional[int] = None    # overrides cfd.max_iter for this job


def case_tag(study: str, profile: str, fineness: float, rn: float, grid: str, mach: float) -> str:
    """File name stem of one CFD point (same convention as the original scripts)."""
    return f"{study}_{profile}_f{fineness:g}_rn{rn:g}_{grid}_M{mach:g}"


def solve_job(job: Job, settings: SolverSettings, cp_dir: str, fields_dir: Optional[str] = None) -> list[dict]:
    """Solve one geometry for its Mach sequence; write one wall-pressure file per Mach number."""
    X, Y, ish = make_grid(job.profile, job.fineness, rn=job.rn, spec=settings.grid_spec(job.grid))
    mesh = Mesh(X, Y, ish)
    s = EulerAxi(mesh, job.machs[0], cfl=settings.cfl)
    max_it = (job.max_iter or settings.max_iter) * (settings.fine_iter_factor if job.grid == "fine" else 1)
    rows = []
    for k, M in enumerate(job.machs):
        t0 = time.time()
        s.set_mach(M)
        try:
            cd = s.run(max_iter=max_it, tol=settings.tolerance, check=settings.check,
                       first_order_iters=settings.first_order_iters if k == 0 else 0)
        except FloatingPointError:
            s = EulerAxi(mesh, M, cfl=settings.fallback_cfl)          # restart from free stream
            cd = s.run(max_iter=max_it, tol=settings.tolerance, check=settings.check,
                       first_order_iters=settings.first_order_iters)
        it = s.history[-1][0]
        rows.append(dict(study=job.study, shape=job.profile.name, fineness=job.fineness, rn=job.rn,
                         grid=job.grid, mach=M, cd=cd, iters=it, converged=int(it < max_it),
                         seconds=round(time.time() - t0, 1)))
        xw, cp = s.wall_cp()
        tag = case_tag(job.study, job.profile.name, job.fineness, job.rn, job.grid, M)
        np.savez_compressed(os.path.join(cp_dir, tag + ".npz"), x=xw, cp=cp,
                            y=0.5 * (mesh.Y[1:, 0] + mesh.Y[:-1, 0]), ish=mesh.i_sh,
                            study=job.study, profile=job.profile.name, fineness=job.fineness, rn=job.rn,
                            grid=job.grid, mach=M)
        if job.save_fields and fields_dir and any(np.isclose(M, m) for m in settings.save_field_machs):
            W = cons2prim(s.U)
            np.savez_compressed(os.path.join(fields_dir, f"{job.profile.name}_rn{job.rn:g}_M{M:g}.npz"),
                                X=X, Y=Y, W=W.astype(np.float32), ish=ish)
    return rows


def _worker(args: tuple) -> list[dict]:
    job, settings, cp_dir, fields_dir = args
    return solve_job(job, settings, cp_dir, fields_dir)


def jobs_from_studies(studies: Iterable[Mapping], profiles: Mapping[str, Profile],
                      only: Optional[Iterable[str]] = None) -> list[Job]:
    """Expand study definitions (profiles x fineness x tip radii) into jobs."""
    only = set(only) if only else None
    jobs = []
    for st in studies:
        if only and st["name"] not in only:
            continue
        for p, f, rn in itertools.product(st["profiles"], st["fineness"], st.get("tip_radius_ratios", [0.0])):
            jobs.append(Job(st["name"], profiles[p], float(f) if float(f) != int(f) else int(f), float(rn),
                            st.get("grid", "medium"), [float(m) for m in st["machs"]],
                            bool(st.get("save_fields", False)),
                            int(st["max_iter"]) if st.get("max_iter") else None))
    return jobs


def run_jobs(jobs: list[Job], settings: SolverSettings, cp_dir: Path, fields_dir: Optional[Path],
             log_path: Optional[Path], workers: int = 4, verbose: bool = True, plugins: Iterable = ()) -> list[dict]:
    """Solve jobs in parallel; append every finished point to the run log.

    `plugins` (files of project.plugins) are loaded in every worker so that custom families exist there.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cp_dir.mkdir(parents=True, exist_ok=True)
    if fields_dir:
        fields_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    args = [(j, settings, str(cp_dir), str(fields_dir) if fields_dir else None) for j in jobs]
    fh = writer = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not log_path.exists()
        fh = open(log_path, "a", newline="")
        writer = csv.DictWriter(fh, LOG_FIELDS)
        if new:
            writer.writeheader()
    try:
        if workers <= 1:
            results = map(_worker, args)
            for rows in results:
                out += _record(rows, writer, fh, verbose)
        else:
            from .plugins import load_plugins
            with Pool(workers, initializer=load_plugins, initargs=(list(plugins),)) as pool:
                for rows in pool.imap_unordered(_worker, args):
                    out += _record(rows, writer, fh, verbose)
    finally:
        if fh:
            fh.close()
    return out


def _record(rows: list[dict], writer, fh, verbose: bool) -> list[dict]:
    if writer:
        for r in rows:
            writer.writerow(r)
        fh.flush()
    if verbose and rows:
        r0 = rows[0]
        print(f"  CFD {r0['study']} {r0['shape']} L/D={r0['fineness']:g} rn/R={r0['rn']:g} {r0['grid']}: "
              + ", ".join(f"M{r['mach']:g}: {r['cd']:.4f} ({r['iters']} it)" for r in rows), flush=True)
    return rows


# -------------------------------------------------------------------------------- database
def rebuild_database(cp_dir: Path, settings: SolverSettings, profiles: Mapping[str, Profile],
                     out_csv: Optional[Path] = None) -> pd.DataFrame:
    """Rebuild the wave-drag table from the wall-pressure files.

    The grid of every geometry is regenerated (deterministic) to get the exact wall-face
    normals, and C_D,w = -2 sum cp n_x l r over the nose faces (R = 1).
    """
    rows, meshes = [], {}
    for fn in sorted(glob.glob(os.path.join(str(cp_dir), "*.npz"))):
        z = np.load(fn)
        if "study" in z.files:
            study, shape, grid = str(z["study"]), str(z["profile"]), str(z["grid"])
            f, rn, M = float(z["fineness"]), float(z["rn"]), float(z["mach"])
        else:
            m = TAG_RE.search(os.path.basename(fn))
            if not m:
                continue
            study, shape, f, rn, grid, M = m.groups()
            f, rn, M = float(f), float(rn), float(M)
        if shape not in profiles:
            print(f"  skipping {os.path.basename(fn)}: profile '{shape}' is not defined")
            continue
        key = (shape, f, rn, grid)
        if key not in meshes:
            X, Y, ish = make_grid(profiles[shape], f, rn=rn, spec=settings.grid_spec(grid))
            meshes[key] = Mesh(X, Y, ish)
        cd = meshes[key].wall_drag(z["cp"])
        rows.append(dict(study=study, shape=shape, fineness=f, rn=rn, grid=grid, mach=M, cd=cd))
    df = pd.DataFrame(rows, columns=["study", "shape", "fineness", "rn", "grid", "mach", "cd"])
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def load_database(path: Path) -> pd.DataFrame:
    """Read the wave-drag database (columns study, shape, fineness, rn, grid, mach, cd)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"wave-drag database not found: {path} (set aerodynamics.database or run "
                                f"'python -m nosecone cfd' and 'rebuild-database')")
    return pd.read_csv(path)


def studies_from_cp_files(cp_dir: Path) -> list[dict]:
    """Study definitions that regenerate exactly the cases present in a cp folder.

    Every geometry's Mach sequence is its stored Mach numbers, highest first; geometries with the
    same sequence are grouped.
    """
    chains: dict[tuple, list] = {}
    for fn in sorted(glob.glob(os.path.join(str(cp_dir), "*.npz"))):
        m = TAG_RE.search(os.path.basename(fn))
        if not m:
            continue
        study, shape, f, rn, grid, M = m.groups()
        chains.setdefault((study, shape, float(f), float(rn), grid), []).append(float(M))
    groups: dict[tuple, list] = {}
    for (study, shape, f, rn, grid), machs in chains.items():
        key = (study, grid, rn, tuple(sorted(machs, reverse=True)))
        groups.setdefault(key, []).append((shape, f))
    out = []
    for (study, grid, rn, machs), geoms in groups.items():
        by_shape: dict[str, list] = {}
        for shape, f in geoms:
            by_shape.setdefault(shape, []).append(f)
        for shape, fs in by_shape.items():
            out.append(dict(name=study, profiles=[shape], fineness=sorted(fs), tip_radius_ratios=[rn],
                            grid=grid, machs=list(machs)))
    return out

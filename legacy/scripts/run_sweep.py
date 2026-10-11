"""Euler sweep: nose pressure-drag coefficient vs shape, fineness, bluntness and Mach.

Each (shape, fineness, bluntness) geometry is solved from the highest Mach number down,
every run starting from the previous converged field (Mach continuation).
Results are appended to results/euler_sweep.csv; wall Cp distributions go to
results/cp/<case>.npz.

usage: python run_sweep.py main|blunt|grid [n_workers]
"""
import os
import sys
import time
import csv
import itertools
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import euler_axi as e  # noqa: E402
from geometry import SHAPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MACHS = [1.6, 1.4, 1.3, 1.2, 1.1]
FIELDS = ["study", "shape", "fineness", "rn", "grid", "mach", "cd", "iters", "converged", "seconds"]
MAX_IT = 6000
GRIDS = {"coarse": (60, 20, 40, 4e-3), "medium": (120, 40, 80, 2e-3), "fine": (240, 80, 160, 1e-3)}


def solve_geometry(args):
    study, shape, f, rn, grid, machs = args
    ni, nc, nj, dy = GRIDS[grid]
    X, Y, ish = e.make_grid(shape, f, rn=rn, ni_nose=ni, ni_cyl=nc, nj=nj, wall_dy=dy)
    mesh = e.Mesh(X, Y, ish)
    s = e.EulerAxi(mesh, machs[0], cfl=0.85)
    rows = []
    for k, M in enumerate(machs):
        t0 = time.time()
        U_prev = s.U.copy()
        s.set_mach(M)
        max_it = 2 * MAX_IT if grid == "fine" else MAX_IT
        try:
            cd = s.run(max_iter=max_it, check=250, first_order_iters=300 if k == 0 else 0)
        except FloatingPointError:
            s = e.EulerAxi(mesh, M, cfl=0.5)          # restart from free stream
            cd = s.run(max_iter=max_it, check=250)
        it = s.history[-1][0]
        conv = it < max_it
        rows.append(dict(study=study, shape=shape, fineness=f, rn=rn, grid=grid, mach=M, cd=cd,
                         iters=it, converged=int(conv), seconds=round(time.time() - t0, 1)))
        xw, cp = s.wall_cp()
        tag = f"{study}_{shape}_f{f:g}_rn{rn:g}_{grid}_M{M:g}"
        np.savez_compressed(os.path.join(RES, "cp", tag + ".npz"), x=xw, cp=cp,
                            y=0.5 * (mesh.Y[1:, 0] + mesh.Y[:-1, 0]), ish=mesh.i_sh)
        _ = U_prev
    return rows


def main():
    which = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(os.path.join(RES, "cp"), exist_ok=True)
    jobs = []
    if which == "main":
        for shape, f in itertools.product(SHAPES, [3, 4, 5, 6]):
            jobs.append(("main", shape, f, 0.0, "medium", MACHS))
    elif which == "fast":
        # restart after the first pass: cone 3-6 and ogive 3-4 are already in the CSV
        M4 = [1.6, 1.4, 1.2, 1.1]
        for shape in ["vonkarman", "power050", "lvhaack", "ogive", "parabolic", "power075"]:
            for f in [3, 4, 5]:
                if shape == "ogive" and f in (3, 4):
                    continue
                jobs.append(("main", shape, f, 0.0, "medium", M4))
            if shape == "power050":
                jobs.append(("grid", "vonkarman", 4, 0.0, "fine", [1.4, 1.2]))
                jobs.append(("grid", "vonkarman", 4, 0.0, "coarse", [1.4, 1.2]))
                for rn in [0.05, 0.1, 0.2]:
                    jobs.append(("blunt", "vonkarman", 4, rn, "medium", [1.6, 1.4, 1.2]))
    elif which == "fast2":
        for shape, f in [("ogive", 5), ("parabolic", 4), ("power075", 4)]:
            jobs.append(("main", shape, f, 0.0, "medium", [1.6, 1.4, 1.2]))
    elif which == "blunt":
        for shape, rn in itertools.product(["vonkarman", "power050"], [0.0, 0.05, 0.1, 0.15, 0.2]):
            jobs.append(("blunt", shape, 4.5, rn, "medium", [1.6, 1.4, 1.2]))
    elif which == "cones":
        for th in [10.0, 15.0, 20.0]:
            jobs.append(("cones", "cone", round(1 / (2 * np.tan(np.radians(th))), 4), 0.0, "medium",
                         [2.0, 1.6, 1.4, 1.2]))
    elif which == "grid":
        for g in ["coarse", "medium", "fine"]:
            jobs.append(("grid", "vonkarman", 4, 0.0, g, [1.4, 1.2]))
            jobs.append(("grid", "power050", 4, 0.1, g, [1.4, 1.2]))
    out = os.path.join(RES, "euler_sweep.csv")
    new = not os.path.exists(out)
    with Pool(nw) as pool, open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for rows in pool.imap_unordered(solve_geometry, jobs):
            for r in rows:
                w.writerow(r)
            fh.flush()
            print(rows[0]["shape"], rows[0]["fineness"], rows[0]["rn"], rows[0]["grid"],
                  [(r["mach"], round(r["cd"], 4), r["iters"]) for r in rows], flush=True)


if __name__ == "__main__":
    main()

"""Euler runs at the design fineness L/D = 4.5 for every shape, saving the full flow field.

Gives (i) C_D,w at L/D = 4.5 computed directly (check of the fineness interpolation) and
(ii) the Mach-number fields for the design-catalogue figure.
Writes results/cp/f45_<shape>_f4.5_rn<rn>_medium_M<M>.npz (wall C_p, as the sweep) and
results/fields/<shape>_rn<rn>_M<M>.npz (cell-centred primitive variables + grid).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402

import euler_axi as e  # noqa: E402
from geometry import SHAPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
F = 4.5
MACHS = [1.6, 1.4, 1.2]
FIELD_MACHS = (1.4, 1.2)


def run(args):
    shape, rn = args
    X, Y, ish = e.make_grid(shape, F, rn=rn)
    mesh = e.Mesh(X, Y, ish)
    s = e.EulerAxi(mesh, MACHS[0], cfl=0.85)
    out = []
    for k, M in enumerate(MACHS):
        s.set_mach(M)
        cd = s.run(max_iter=6000, check=250, first_order_iters=300 if k == 0 else 0)
        xw, cp = s.wall_cp()
        np.savez_compressed(os.path.join(RES, "cp", f"f45_{shape}_f{F:g}_rn{rn:g}_medium_M{M:g}.npz"),
                            x=xw, cp=cp, y=0.5 * (mesh.Y[1:, 0] + mesh.Y[:-1, 0]), ish=mesh.i_sh)
        if M in FIELD_MACHS:
            W = e.cons2prim(s.U)
            np.savez_compressed(os.path.join(RES, "fields", f"{shape}_rn{rn:g}_M{M:g}.npz"),
                                X=X, Y=Y, W=W.astype(np.float32), ish=ish)
        out.append((M, round(float(cd), 5), s.history[-1][0]))
    return shape, rn, out


if __name__ == "__main__":
    os.makedirs(os.path.join(RES, "fields"), exist_ok=True)
    jobs = [(s, 0.0) for s in SHAPES] + [("vonkarman", 0.05)]
    with Pool(8) as pool:
        for shape, rn, out in pool.imap_unordered(run, jobs):
            print(shape, rn, out, flush=True)

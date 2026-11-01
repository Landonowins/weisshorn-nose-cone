"""Run density-based 720 mm cases one after the other on the session behind server_info-A.txt.

Each case: read its mesh, apply the setup of setup_fluent.py at the given free stream, hybrid
initialisation, A_adiab's schedule (to iteration 4000), then 500-iteration blocks until cd-nose
changes by less than 0.1 % over the last 500 iterations (study A rule), then save case/data and
the wall data of wall_nose + wall_coupler.

Usage: python run_queue.py RUN [RUN ...]   with RUN in CASES below
"""
import sys

import setup_fluent as F

HERE = F.HERE
# free stream at M 1.20 from fluent_points_python.csv (design A row, used for both designs, as at burnout)
M12 = dict(p_inf=84801.4, m_inf=1.20, t_inf=278.554)
M095 = dict(p_inf=90714.2, m_inf=0.95, t_inf=282.149)
CASES = {  # mesh, free stream, solver (M 0.95: pressure-based, the density-based solver did not converge in study A)
    "A_M1.20": ("A_mesh_v3.msh", M12, "density"),
    "B_M1.20": ("B_mesh_v2.msh", M12, "density"),
    "A_M0.95": ("A_mesh_v3.msh", M095, "pressure"),
    "B_M0.95": ("B_mesh_v2.msh", M095, "pressure"),
    "B_T300": ("B_adiab", None, "t300"),  # walls at 300 K, restarted from the converged adiabatic run
}


def cd_nose_change(run, window=500):
    rows = {}
    for line in open(HERE / f"{run}_cd.out"):
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows[int(parts[0])] = float(parts[2])
    its = sorted(rows)[-window:]
    vals = [rows[i] for i in its]
    return its[-1], vals[-1], 100 * (max(vals) - min(vals)) / (sum(vals) / len(vals))


def converge(s, run, block=500, max_blocks=8):
    it, cd, ch = cd_nose_change(run)
    for _ in range(max_blocks):
        if ch < 0.1:
            break
        s.settings.solution.run_calculation.iterate(iter_count=block)
        it, cd, ch = cd_nose_change(run)
        print(f"{run}: iteration {it}, cd-nose {cd:.5f}, change over last 500 {ch:.3f} %", flush=True)
    return it, cd, ch


def export_wall(s, run):
    s.tui.file.export.ascii((HERE / f"{run}_wall.csv").as_posix(), "wall_nose", "wall_coupler", "()", "yes",
                            "pressure", "pressure-coefficient", "temperature", "heat-flux", "y-plus",
                            "skin-friction-coef", "wall-shear", "()", "yes")


def finish(s, run):
    """Converge, save and export the case already in the session (wall forces go to the transcript)."""
    it, cd, ch = converge(s, run)
    s.settings.file.write(file_type="case-data", file_name=(HERE / run).as_posix())
    export_wall(s, run)
    s.tui.report.forces.wall_forces("yes", "1", "0", "no")
    print(f"DONE {run}: iteration {it}, cd-nose {cd:.5f}, change over last 500 {ch:.3f} %", flush=True)


def run_case(s, run):
    mesh, fs, solver = CASES[run]
    if solver == "t300":  # as A_T300: 3500 iterations from the adiabatic solution, then the cd-nose rule
        s.settings.file.read_case_data(file_name=(HERE / f"{mesh}.cas.h5").as_posix())
        F.t300(s, run=run)
        s.settings.solution.run_calculation.iterate(iter_count=3500)
        finish(s, run)
        return
    s.settings.file.read_case(file_name=(HERE / mesh).as_posix())
    F.general(s)
    if solver == "pressure":
        F.pressure_based(s)
    F.bcs(s, run=run, **fs)
    if solver == "pressure":
        F.pb_order(s, "first")
        F.init(s, run=run)
        F.schedule_pb(s, run)
    else:
        F.methods(s)
        F.init(s, run=run)
        F.schedule(s, run)
    finish(s, run)


if __name__ == "__main__":
    s = F.connect()
    for run in sys.argv[1:]:
        if run.startswith("finish:"):
            finish(s, run.split(":", 1)[1])
        else:
            run_case(s, run)

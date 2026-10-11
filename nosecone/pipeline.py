"""Single entry point that runs the whole chain for one configuration.

    database -> (missing CFD cases) -> calibration -> sweep -> sensitivities -> robustness
    -> trajectories -> heating -> bluntness -> design definition -> decision matrix
    -> CAD export -> plots -> tables / LaTeX -> summary
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import cfd
from . import config as config_mod
from . import export
from .aerodynamics import missing_cases
from .config import Config
from .design import Study
from .geometry import make_profiles
from .reporting import summary_markdown, write_latex, write_text_results


class Logger:
    """Prints to the console and to <output>/run.log."""

    def __init__(self, path: Path, quiet: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")
        self.quiet = quiet

    def __call__(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        """Close the log file."""
        self.fh.close()


def compute_missing_cases(cfg: Config, db: pd.DataFrame, log: Callable[[str], None]) -> pd.DataFrame:
    """Run the Euler solver for profiles / tip radii of the sweep that have no CFD data; extend the database."""
    n = cfg.nose
    need = missing_cases(db, cfg, list(dict.fromkeys(list(n.profiles) + [n.baseline_profile, n.design_profile])),
                         sorted(set(list(n.fineness) + [n.baseline_fineness, n.design_fineness])), n.tip_radius_ratios)
    if not need:
        return db
    profiles = make_profiles(n.custom_profiles)
    geoms = sorted({(p, rn) for p, _, rn in need})
    fs = sorted(set(n.fineness) | {n.baseline_fineness, n.design_fineness})
    # the wave-drag model needs at least two fineness ratios to interpolate: use the sweep's extremes and the design
    f_run = sorted({min(fs), max(fs), n.design_fineness})
    jobs = [cfd.Job(cfg.aerodynamics.studies[0], profiles[p], f, rn, cfg.aerodynamics.grid,
                    list(cfg.aerodynamics.missing_machs)) for p, rn in geoms for f in f_run]
    log(f"Computing {len(jobs)} missing CFD geometries ({', '.join(f'{p} rn/R={rn:g}' for p, rn in geoms)}) "
        f"with {cfg.cfd.workers} workers; this takes several minutes per geometry...")
    settings = cfd.SolverSettings.from_config(cfg)
    cfd.run_jobs(jobs, settings, cfg.path(cfg.cfd.cp_dir), None, cfg.path(cfg.cfd.log), cfg.cfd.workers,
                 plugins=cfg.project.plugins)
    new = cfd.rebuild_database(cfg.path(cfg.cfd.cp_dir), settings, profiles)
    keys = ["study", "shape", "fineness", "rn", "grid", "mach"]
    db = pd.concat([db, new]).drop_duplicates(keys, keep="last").reset_index(drop=True)
    a = cfg.aerodynamics
    if a.user_database:
        target = cfg.path(a.user_database)
        main = cfd.load_database(cfg.path(a.database))
        extra = pd.concat([db, main, main]).drop_duplicates(keys, keep=False)   # rows not in the main database
    else:
        target, extra = cfg.path(a.database), db
    target.parent.mkdir(parents=True, exist_ok=True)
    extra.to_csv(target, index=False)
    log(f"  database updated: {target}")
    return db


def load_databases(cfg: Config) -> pd.DataFrame:
    """Main CFD database plus the optional user database."""
    a = cfg.aerodynamics
    db = cfd.load_database(cfg.path(a.database))
    if a.user_database and cfg.path(a.user_database).exists():
        keys = ["study", "shape", "fineness", "rn", "grid", "mach"]
        user = pd.read_csv(cfg.path(a.user_database))
        db = pd.concat([db, user]).drop_duplicates(keys, keep="last").reset_index(drop=True)
    return db


def export_cad(st: Study, outdir: Path, log: Callable[[str], None]) -> dict:
    """Profile CSV, STL solids (design and candidates) and the STEP solid of the design."""
    cfg, c = st.cfg, st.cfg.cad
    outdir.mkdir(parents=True, exist_ok=True)
    p, f, rnr = st.design
    pr, L, R = st.profiles[p], st.L(f), st.R
    rn = rnr * R
    sh = cfg.nose.shoulder_length
    res: dict = {}
    if c.profile_csv:
        path = outdir / "nose_profile_final.csv"
        res["profile_csv"] = path
        res["physical_length_mm"] = export.write_profile_csv(path, pr, L, R, rn, c.profile_points, c.profile_clustering)
    if c.stl:
        if c.candidates:
            for name in cfg.nose.profiles:
                x, r = export.outline(st.profiles[name], L, R, sh, n=c.stl_points)
                export.write_stl(outdir / f"nose_{name}_LD{f:g}.stl", x, r, n_az=c.stl_facets, name=f"{name} L/D={f}")
        x, r = export.outline(pr, L, R, sh, rn=rn, n=c.stl_points_design)
        path = outdir / f"nose_{p}_LD{f:g}_rn{1e3 * rn:g}mm_FINAL.stl"
        export.write_stl(path, x, r, n_az=c.stl_facets, name=f"{cfg.project.name} nose cone FINAL")
        chk = export.stl_check(path)
        res["stl_check"] = chk
        log(f"  STL {path.name}: {chk['triangles']} triangles, open edges {chk['open_edges']}, "
            f"volume {chk['volume_mm3'] / 1e6:.3f} L")
    if c.step:
        if export.step_available():
            xs, rs = export.profile_points(pr, L, R, rn, c.step_points, c.profile_clustering)
            path = outdir / f"nose_{p}_LD{f:g}_rn{1e3 * rn:g}mm_FINAL.step"
            chk = export.write_step(path, 1e3 * xs, 1e3 * rs, 1e3 * sh, name=p)
            res["step_check"] = chk
            log(f"  STEP {path.name}: {chk['solids']} solid, volume {chk['volume_mm3'] / 1e6:.3f} L")
        else:
            log("  STEP skipped: the optional package gmsh is not installed (pip install gmsh)")
    return res


def run(cfg: Config, plots: Optional[list] = None, latex: Optional[bool] = None, quiet: bool = False) -> dict:
    """Run the full study for a configuration and write all outputs to cfg.out.

    Args:
        cfg: validated configuration.
        plots: plots to draw (overrides output.plots); [] for none.
        latex: write LaTeX tables (overrides output.latex).
        quiet: do not print progress to the console (the log file is still written).

    Returns:
        Dictionary with the study object and every result table.
    """
    t0 = time.time()
    out = cfg.out
    dirs = {k: out / k for k in ("tables", "figures", "cad", "latex")}
    for d in ("tables", "figures", "cad"):
        dirs[d].mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run.log", quiet)
    config_mod.save(cfg, out / "config_used.yaml")
    log(f"== {cfg.project.name}: nose-cone study -> {out}")
    db = load_databases(cfg)
    log(f"CFD database: {len(db)} points ({cfg.path(cfg.aerodynamics.database).name}"
        + (f" + {cfg.path(cfg.aerodynamics.user_database).name}" if cfg.aerodynamics.user_database else "") + ")")
    if cfg.aerodynamics.compute_missing:
        db = compute_missing_cases(cfg, db, log)

    st = Study(cfg, db)
    log(f"Drag factor k = {st.k:.4f} ({'calibrated: baseline reaches ' + format(cfg.flight.target_apogee, '.0f') + ' m' if cfg.flight.calibrate else 'fixed'})")

    log("Sweep: " + f"{len(cfg.nose.profiles)} profiles x {len(cfg.nose.fineness)} fineness x {len(cfg.nose.tip_radius_ratios)} tip radii")
    trade = st.sweep(log=log)
    trade.to_csv(dirs["tables"] / "trade_table.csv", index=False)
    sens = st.sensitivities()
    write_text_results(st, dirs["tables"], sens)
    log(f"  sensitivities: {sens['dz_dm']:.1f} m/kg, {sens['dz_per_cd_step']:.1f} m per {sens['cd_step']:g} of C_D")

    robust = None
    if cfg.robustness.enabled:
        log(f"Robustness at L/D = {cfg.robustness.fineness:g} (relative to {cfg.robustness.reference_profile}):")
        robust = st.robustness(verbose=True, log=log)
        robust.to_csv(dirs["tables"] / "robustness.csv", index=False)

    traj = st.trajectories()
    _, nom, env = traj
    log(f"Trajectories: M_max {nom['M'].max():.2f} nominal, {env['M'].max():.2f} envelope; apogee {nom['apogee']:.0f} m")

    heat = None
    if cfg.heating.enabled:
        heat = st.heating(nom, env)
        log(f"Heating [degC] nominal/envelope: T0 {heat['T0_peak'][0]:.0f}/{heat['T0_peak'][1]:.0f}, metal tip "
            f"{heat['tip_insert'][0]:.0f}/{heat['tip_insert'][1]:.0f}, printed tip {heat['printed_tip'][0]:.0f}/"
            f"{heat['printed_tip'][1]:.0f}, shell inner {heat['shell_inner'][0]:.0f}/{heat['shell_inner'][1]:.0f}")
        (dirs["tables"] / "heating.json").write_text(json.dumps(heat, indent=2))

    blunt = []
    if cfg.bluntness.enabled:
        blunt = st.bluntness()
        if blunt:
            pd.DataFrame(blunt).to_csv(dirs["tables"] / "bluntness.csv", index=False)
            for b in blunt:
                log(f"Tip r_n/R = {b['rn_over_R']:g}: wave drag {b['drag_cost_pct']:+.1f} %, apogee {b['d_apogee']:+.0f} m")
        else:
            log("Bluntness: no data for the configured study/profile, skipped")

    design = st.design_definition(trade)
    pd.Series({k: v for k, v in design.items() if k != "trade_row"}).to_csv(dirs["tables"] / "design.csv", header=["value"])

    matrix = None
    if cfg.trade_matrix.enabled and np.isclose(trade.fineness, cfg.trade_matrix.fineness).any():
        matrix = st.trade_matrix(trade)
        if matrix is not None:
            matrix.to_csv(dirs["tables"] / "trade_matrix.csv")
    if cfg.payload.enabled:
        st.catalogue(trade, tuple(x for x in sorted(set(trade.fineness)) if x >= cfg.nose.baseline_fineness - 1e-9)) \
            .to_csv(dirs["tables"] / "catalogue.csv")

    log("CAD export:")
    cad = export_cad(st, dirs["cad"], log)

    plot_list = cfg.output.plots if plots is None else plots
    if plot_list:
        from .plotting import figures as F
        from .plotting import style
        names = F.ALL_PLOTS if "all" in plot_list else plot_list
        tr = cfg.output.text_renderer
        used = style.setup(None if tr == "auto" else tr == "latex")
        ctx = F.PlotContext(cfg, st, trade, robust, traj, dirs["figures"], log)
        log(f"Plots ({'LaTeX' if used else 'mathtext'} text): {', '.join(names)}")
        for name in names:
            if name not in F.PLOTS:
                log(f"  unknown plot '{name}' (known: {', '.join(F.ALL_PLOTS)})")
                continue
            if name == "robustness" and robust is None:
                continue
            try:
                F.PLOTS[name](ctx)
            except Exception as exc:  # a failed plot must not stop the run
                log(f"  plot '{name}' failed: {type(exc).__name__}: {exc}")

    if cfg.output.latex if latex is None else latex:
        files = write_latex(st, dirs["latex"], trade, matrix, blunt, heat, design)
        log(f"LaTeX: {', '.join(p.name for p in files)}")

    summary = summary_markdown(st, trade, sens, robust, heat, blunt, design, matrix)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    log(f"Summary: {out / 'summary.md'}   ({time.time() - t0:.0f} s)")
    log.close()
    return dict(study=st, trade=trade, sensitivities=sens, robustness=robust, trajectories=traj, heating=heat,
                bluntness=blunt, design=design, matrix=matrix, cad=cad)

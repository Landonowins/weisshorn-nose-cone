"""Command-line interface: ``python -m nosecone <command> [options]``.

Commands:
    run CONFIG               full study (sweep, robustness, heating, CAD, plots, tables)
    export CONFIG            CAD files of the design only (profile CSV, STL, STEP)
    cfd CONFIG               run the Euler studies listed in cfd.studies (slow)
    rebuild-database CONFIG  rebuild the wave-drag CSV from the CFD wall-pressure files
    validate CONFIG          check a configuration file and print the resolved values
    describe-config          list every parameter (unit, valid range, default, meaning)
    profiles                 list the profile families and the built-in profiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _load(path: str):
    from .config import ConfigError, load
    try:
        return load(path)
    except ConfigError as e:
        sys.exit(f"configuration error: {e}")


def cmd_run(a: argparse.Namespace) -> None:
    """``run``: full study."""
    from .pipeline import run
    cfg = _load(a.config)
    if a.output:
        cfg.project.output_dir = str(Path(a.output).resolve())
    plots = None
    if a.no_plots:
        plots = []
    elif a.plots:
        plots = [p.strip() for p in a.plots.split(",") if p.strip()]
    run(cfg, plots=plots, latex=True if a.latex else None, quiet=a.quiet)


def cmd_export(a: argparse.Namespace) -> None:
    """``export``: CAD files of the design only."""
    from .cfd import load_database
    from .design import Study
    from .pipeline import export_cad
    cfg = _load(a.config)
    st = Study(cfg, load_database(cfg.path(cfg.aerodynamics.database)), k=1.0)
    out = cfg.out / "cad"
    export_cad(st, out, print)
    print(f"CAD files in {out}")


def cmd_cfd(a: argparse.Namespace) -> None:
    """``cfd``: run the Euler studies of cfd.studies."""
    from . import cfd
    from .geometry import make_profiles
    cfg = _load(a.config)
    profiles = make_profiles(cfg.nose.custom_profiles)
    jobs = cfd.jobs_from_studies(cfg.cfd.studies, profiles, a.study)
    if not jobs:
        sys.exit("no CFD jobs: define cfd.studies in the configuration (or check --study)")
    n_pts = sum(len(j.machs) for j in jobs)
    print(f"{len(jobs)} geometries, {n_pts} Mach points, {a.workers or cfg.cfd.workers} workers")
    if a.dry_run:
        for j in jobs:
            print(f"  {j.study:8s} {j.profile.name:12s} L/D={j.fineness:<6g} rn/R={j.rn:<5g} {j.grid:7s} M={j.machs}")
        return
    cp_dir = Path(a.cp_dir).resolve() if a.cp_dir else cfg.path(cfg.cfd.cp_dir)
    fields = Path(a.fields_dir).resolve() if a.fields_dir else cfg.path(cfg.cfd.fields_dir)
    log = Path(a.log).resolve() if a.log else cfg.path(cfg.cfd.log)
    cfd.run_jobs(jobs, cfd.SolverSettings.from_config(cfg), cp_dir, fields, log, a.workers or cfg.cfd.workers,
                 plugins=cfg.project.plugins)
    print(f"done; wall-pressure files in {cp_dir}. Run 'rebuild-database' to update the wave-drag table.")


def cmd_rebuild(a: argparse.Namespace) -> None:
    """``rebuild-database``: wave-drag CSV from the wall-pressure files."""
    from . import cfd
    from .geometry import make_profiles
    cfg = _load(a.config)
    cp_dir = Path(a.cp_dir).resolve() if a.cp_dir else cfg.path(cfg.cfd.cp_dir)
    out = Path(a.output).resolve() if a.output else cfg.path(cfg.aerodynamics.database)
    df = cfd.rebuild_database(cp_dir, cfd.SolverSettings.from_config(cfg), make_profiles(cfg.nose.custom_profiles), out)
    print(f"{len(df)} points written to {out}")


def cmd_validate(a: argparse.Namespace) -> None:
    """``validate``: check a configuration file."""
    import yaml
    from .config import to_dict
    cfg = _load(a.config)
    print(f"{a.config}: OK")
    if a.show:
        print(yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True))


def cmd_describe(a: argparse.Namespace) -> None:
    """``describe-config``: list every parameter."""
    from .config import describe
    rows = describe(include_advanced=not a.basic)
    if a.latex:
        from .docs import config_tables_latex
        print(config_tables_latex(rows))
        return
    sec = None
    for r in rows:
        if r["section"] != sec:
            sec = r["section"]
            print(f"\n[{sec}]")
        adv = " (advanced)" if r["advanced"] else ""
        print(f"  {r['name']} = {r['default']!r}  [{r['unit']}]  {('valid: ' + r['valid']) if r['valid'] else ''}{adv}")
        print(f"      {r['doc']}")


def cmd_profiles(a: argparse.Namespace) -> None:
    """``profiles``: list families and built-in profiles."""
    from .geometry import BUILTIN, describe_families
    print("Families (use in nose.custom_profiles as family: <name>):")
    for name, params, doc in describe_families():
        ps = ", ".join(f"{k}={v:g}" for k, v in params.items()) or "no parameter"
        print(f"  {name:14s} ({ps})  {doc}")
    print("\nBuilt-in profiles (names usable in nose.profiles):")
    for p in BUILTIN.values():
        ps = ", ".join(f"{k}={v:.4g}" for k, v in p.params)
        print(f"  {p.name:10s} family {p.family:14s} {ps}")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the command."""
    ap = argparse.ArgumentParser(prog="python -m nosecone", description="Parametric nose-cone design tool (ERT).")
    ap.add_argument("--version", action="version", version=f"nosecone {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run the full study for a configuration")
    p.add_argument("config")
    p.add_argument("--output", help="output folder (overrides project.output_dir)")
    p.add_argument("--plots", help="comma-separated plot names (overrides output.plots)")
    p.add_argument("--no-plots", action="store_true", help="skip all plots")
    p.add_argument("--latex", action="store_true", help="also write LaTeX tables and macros")
    p.add_argument("--quiet", action="store_true", help="only write the log file")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("export", help="write the CAD files of the design only")
    p.add_argument("config")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("cfd", help="run the Euler studies of cfd.studies")
    p.add_argument("config")
    p.add_argument("--study", action="append", help="only this study (repeatable)")
    p.add_argument("--workers", type=int, help="parallel processes (overrides cfd.workers)")
    p.add_argument("--cp-dir", help="output folder of the wall-pressure files (overrides cfd.cp_dir)")
    p.add_argument("--fields-dir", help="output folder of the flow fields (overrides cfd.fields_dir)")
    p.add_argument("--log", help="run log CSV (overrides cfd.log)")
    p.add_argument("--dry-run", action="store_true", help="only list the jobs")
    p.set_defaults(func=cmd_cfd)
    p = sub.add_parser("rebuild-database", help="rebuild the wave-drag CSV from the wall-pressure files")
    p.add_argument("config")
    p.add_argument("--cp-dir", help="folder of the wall-pressure files (overrides cfd.cp_dir)")
    p.add_argument("--output", help="CSV to write (overrides aerodynamics.database)")
    p.set_defaults(func=cmd_rebuild)
    p = sub.add_parser("validate", help="check a configuration file")
    p.add_argument("config")
    p.add_argument("--show", action="store_true", help="print the complete resolved configuration")
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("describe-config", help="list every parameter")
    p.add_argument("--basic", action="store_true", help="hide advanced parameters")
    p.add_argument("--latex", action="store_true", help="LaTeX tables (used for the user guide)")
    p.set_defaults(func=cmd_describe)
    p = sub.add_parser("profiles", help="list profile families and built-in profiles")
    p.set_defaults(func=cmd_profiles)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

"""Check that the tool reproduces the Weisshorn design report (2026_C_FD_REPORT-1_NOSE_CONE).

    python tests/verify_reference.py              # numbers, tables, CAD (about 2 minutes)
    python tests/verify_reference.py --figures    # + pixel comparison of every figure (needs LaTeX)
    python tests/verify_reference.py --cfd        # + three Euler cases recomputed bit for bit (~3 minutes)

The reference files in tests/reference/ are the outputs of the original scripts that produced
the report. Exit code 0 if everything matches (known, documented differences excepted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference"
sys.path.insert(0, str(ROOT))

from nosecone import cfd, config, pipeline  # noqa: E402
from nosecone.design import Study  # noqa: E402
from nosecone.geometry import make_profiles  # noqa: E402

LEGACY_K = 0.5601
"""The original robustness, macros and figure scripts re-read k from calibration.txt (4 decimals)."""

KNOWN = {
    "tab_check45.tex": "parabolic row: source label corrected from 'interp. 4--5' to 'scaled from 4' "
                       "(the data used are L/D 3 and 4; the value at 4.5 is held from 4). Numbers identical.",
}

failures: list[str] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same_text(a: Path, b: Path) -> bool:
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def report(name: str, ok: bool, note: str = "") -> None:
    if ok:
        print(f"  IDENTICAL  {name}")
    elif name in KNOWN:
        print(f"  KNOWN DIFF {name}: {KNOWN[name]}")
    else:
        print(f"  DIFFERS    {name} {note}")
        failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="also compare the figures pixel by pixel")
    ap.add_argument("--cfd", action="store_true", help="also recompute three Euler cases")
    a = ap.parse_args()
    cfg = config.load(ROOT / "configs" / "weisshorn.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="nosecone_verify_"))
    cfg.project.output_dir = str(tmp)
    print(f"Running the Weisshorn configuration into {tmp} ...")
    res = pipeline.run(cfg, plots=[], latex=True, quiet=True)

    print("Tables and numbers (calibrated k):")
    for f in ("trade_table.csv", "calibration.txt", "sensitivity.txt"):
        report(f, same_text(tmp / "tables" / f, REF / f))
    for f in ("macros.tex", "tab_trade.tex", "tab_matrix.tex", "tab_catalogue.tex", "tab_check45.tex"):
        report(f, same_text(tmp / "latex" / f, REF / "latex" / f))
    print("Robustness (k = 0.5601, as in the original script):")
    st = Study(cfg, res["study"].db, k=LEGACY_K)
    rob = st.robustness(verbose=False)
    rob_path = tmp / "tables" / "robustness_legacy_k.csv"
    rob.to_csv(rob_path, index=False)
    report("robustness.csv", same_text(rob_path, REF / "robustness.csv"))

    print("CAD:")
    report("nose_profile_final.csv", same_text(tmp / "cad" / "nose_profile_final.csv", REF / "nose_profile_final.csv"))
    sums = json.loads((REF / "stl_sha256.json").read_text())
    for name, h in sums.items():
        report(name, sha(tmp / "cad" / name) == h)

    print("CFD database rebuilt from the wall-pressure files:")
    s = cfd.SolverSettings.from_config(cfg)
    db = cfd.rebuild_database(cfg.path(cfg.cfd.cp_dir), s, make_profiles(), tmp / "euler_results.csv")
    report("euler_results.csv", same_text(tmp / "euler_results.csv", cfg.path(cfg.aerodynamics.database)))
    _ = db

    if a.figures:
        from PIL import Image
        print("Figures (k = 0.5601 for the figures that the original script drew with it):")
        cfg2 = config.load(ROOT / "configs" / "weisshorn.yaml")
        cfg2.project.output_dir = str(tmp / "fig_legacy_k")
        cfg2.flight.calibrate, cfg2.flight.drag_factor = False, LEGACY_K
        pipeline.run(cfg2, latex=False, quiet=True)
        cfg3 = config.load(ROOT / "configs" / "weisshorn.yaml")
        cfg3.project.output_dir = str(tmp / "fig_full_k")
        pipeline.run(cfg3, plots=["trade"], latex=False, quiet=True)   # drawn from the calibrated trade table
        for ref in sorted((REF / "figures").glob("*.png")):
            src = (tmp / "fig_full_k" / "figures" if ref.stem == "fig_trade" else tmp / "fig_legacy_k" / "figures") / ref.name
            A = np.asarray(Image.open(src).convert("RGB"), dtype=int)
            B = np.asarray(Image.open(ref).convert("RGB"), dtype=int)
            if A.shape != B.shape:
                report(ref.name, False, f"(size {A.shape} vs {B.shape})")
                continue
            d = np.abs(A - B).max(-1)
            n = int((d > 0).sum())
            if n == 0:
                report(ref.name, True)
            elif n <= 5 and d.max() <= 2:
                print(f"  ~IDENTICAL {ref.name}: {n} pixel(s) differ by at most {d.max()}/255 (sub-pixel text position)")
            else:
                report(ref.name, False, f"({n} pixels, max {d.max()})")

    if a.cfd:
        print("Euler solver (three cases, compared bit for bit with data/cfd/cp):")
        profiles = make_profiles()
        jobs = [cfd.Job("grid", profiles["vonkarman"], 4, 0.0, "coarse", [1.4, 1.2]),
                cfd.Job("main", profiles["vonkarman"], 4, 0.0, "medium", [1.6, 1.4]),
                cfd.Job("blunt", profiles["vonkarman"], 4, 0.05, "medium", [1.6])]
        out = tmp / "cp"
        cfd.run_jobs(jobs, s, out, None, None, workers=3, verbose=False)
        for f in sorted(out.glob("*.npz")):
            z1, z0 = np.load(f), np.load(cfg.path(cfg.cfd.cp_dir) / f.name)
            report(f.name, all(np.array_equal(z1[k], z0[k]) for k in ("x", "cp", "y", "ish")))

    print()
    if failures:
        print(f"{len(failures)} unexpected difference(s): {', '.join(failures)}")
        return 1
    print("All outputs match the design report (known differences listed above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

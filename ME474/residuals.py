"""Residual history of every run, from the queue logs (Fluent prints its residual table for every iteration).

Usage: python residuals.py [runs/queue_*.txt ...]      (default: every runs/queue_*.txt)

Writes runs/residuals.csv, one row per run, and prints it. For each equation:
  start   the largest residual of the first 5 iterations (Fluent normalises continuity by it; the other
          equations are scaled residuals, started from the hybrid initialisation)
  final   the residual of the last iteration
  drop    log10(start / final): orders of magnitude down over the run (target: 3, PLAN.md checks)
  creep   log10(final / minimum of the second-order part): how far the residual has come back up from its lowest
          value since the switch to second order (iteration 200 pressure-based, 1000 density-based; 0 = still at it)
A run that appears in several logs (a restart) takes the log where it says DONE.
"""
import csv
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
EQS = ("continuity", "x-velocity", "y-velocity", "energy", "k", "omega")
SWITCH = {"pressure": 200, "density": 1000}  # last first-order iteration of the schedule (run_queue.py)
RUN_HDR = re.compile(r"^=== (\S+): .*, (pressure|density)-based")
ROW = re.compile(r"^\s*(\d+)" + r"\s+(\d\.\d+e[+-]\d+)" * len(EQS) + r"\s+\d+:\d+:\d+\s+\d+\s*$")


def parse(log):
    runs, cur = {}, None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = RUN_HDR.match(line)
        if m:
            cur = runs.setdefault(m.group(1), {"solver": m.group(2), "rows": {}, "done": False, "log": log.name})
            continue
        if cur is None:
            continue
        if line.startswith("DONE "):
            cur["done"] = True
        m = ROW.match(line)
        if m:
            cur["rows"][int(m.group(1))] = [float(v) for v in m.groups()[1:]]
    return runs


def summarise(name, run):
    its = sorted(run["rows"])
    hist = [run["rows"][i] for i in its]
    row = {"run": name, "solver": run["solver"], "iterations": its[-1], "log": run["log"]}
    worst = math.inf
    for j, eq in enumerate(EQS):
        col = [h[j] for h in hist]
        start, final = max(col[:5]), col[-1]
        low = min(c for i, c in zip(its, col) if i > SWITCH[run["solver"]])
        drop = math.log10(start / final)
        row[f"{eq}_final"] = f"{final:.3e}"
        row[f"{eq}_drop"] = f"{drop:.2f}"
        row[f"{eq}_creep"] = f"{math.log10(final / low):.2f}"
        worst = min(worst, drop)
    row["min_drop"] = f"{worst:.2f}"
    return row


def main(args):
    logs = [Path(a) for a in args] or sorted(RUNS.glob("queue_*.txt"))
    runs = {}
    for log in logs:
        for name, run in parse(log).items():
            if run["rows"] and (name not in runs or run["done"]):
                runs[name] = run
    rows = [summarise(n, r) for n, r in runs.items() if r["done"]]
    cols = ["run", "solver", "iterations", "min_drop"] + [f"{e}_{k}" for e in EQS for k in ("drop", "creep", "final")] + ["log"]
    with open(RUNS / "residuals.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{'run':28s} {'its':>5s} " + " ".join(f"{e[:6]:>12s}" for e in EQS) + "   (orders down / creep back up)")
    for r in rows:
        print(f"{r['run']:28s} {r['iterations']:5d} " + " ".join(f"{r[e + '_drop']:>6s}/{r[e + '_creep']:<5s}" for e in EQS))
    print(f"-> {RUNS / 'residuals.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])

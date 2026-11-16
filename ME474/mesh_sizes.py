"""Cells, nodes and faces of every mesh, read from the case files (no Fluent needed).

Usage, with Ansys's Python (it has h5py):
    "C:/Program Files/ANSYS/v242/commonfiles/CPython/3_10/winx64/Release/python/python.exe" ME474/mesh_sizes.py
Writes runs/mesh_sizes.csv. Each count is the sum over the zones of /meshes/1/{cells,nodes,faces}/zoneTopology in
<run>.cas.h5; the cell count is checked against runs/summary.csv.
"""
import csv
from pathlib import Path

import h5py

RUNS = Path(__file__).resolve().parent / "runs"


def count(group):
    z = group["zoneTopology"]
    return int(sum(int(b) - int(a) + 1 for a, b in zip(z["minId"][:], z["maxId"][:])))


def main():
    rows = list(csv.DictReader(open(RUNS / "summary.csv")))
    sizes, nruns = {}, {}
    for r in rows:
        nruns[r["mesh"]] = nruns.get(r["mesh"], 0) + 1
        case = RUNS / r["run"] / f"{r['run']}.cas.h5"
        if r["mesh"] in sizes or not case.exists():
            continue
        with h5py.File(case, "r") as f:
            m = f["meshes/1"]
            sizes[r["mesh"]] = (count(m["cells"]), count(m["nodes"]), count(m["faces"]))
        assert sizes[r["mesh"]][0] == int(r["cells"]), (r["run"], sizes[r["mesh"]], r["cells"])
    with open(RUNS / "mesh_sizes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mesh", "cells", "nodes", "faces", "runs"])
        for mesh in sorted(sizes):
            w.writerow([mesh, *sizes[mesh], nruns[mesh]])
            print(mesh, *sizes[mesh])


if __name__ == "__main__":
    main()

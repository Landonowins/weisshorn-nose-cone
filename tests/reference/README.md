# Reference outputs

Outputs of the original scripts (`legacy/scripts/`) that produced the Weisshorn design report
`2026_C_FD_REPORT-1_NOSE_CONE`. `tests/verify_reference.py` compares the package against them.

| File | Produced by (legacy) | Compared with |
|---|---|---|
| `trade_table.csv`, `calibration.txt`, `sensitivity.txt` | `analysis.py` | `tables/` of a run (calibrated k) |
| `robustness.csv` | `robustness.py` (k re-read from calibration.txt, 4 decimals: 0.5601) | `Study.robustness()` with k = 0.5601 |
| `latex/*.tex` | `make_tables.py` | `latex/` of a run |
| `nose_profile_final.csv` | `analysis.export_profile()` | `cad/` of a run |
| `stl_sha256.json` | `cad.py` (see note) | `cad/*.stl` of a run |
| `figures/*.png` | `figures.py`, `figures_extra.py`, `cad.py` | `--figures` |

Known, intentional differences:

* `latex/tab_check45.tex`: the parabolic row's source label reads `scaled from 4` (the data used are
  L/D 3 and 4 and the value at 4.5 is held from 4); the original said `interp. 4--5`. Numbers identical.
* `nose_ogive_LD4.5.stl`: the checksum is that of the corrected file. Round-off left the ogive apex at
  r ~ 1e-16 m (a 4e-13 mm pin-hole, 256 open edges); radii below 1 nm are now snapped to 0. The 128
  apex triangles moved by at most 4.4e-13 mm; the other seven STL files are byte-identical.
* `fig_cdw.png`: one pixel differs by 1/255 (sub-pixel position of a label).

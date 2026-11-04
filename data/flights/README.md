# Measured flights for the validation of the flight model

Used by `studies/flight_validation.py`; written by `studies/flight_data_prep.py` from the
[RocketPy](https://github.com/RocketPy-Team/RocketPy) repository (MIT licence), commit
`9bd6ad3af8f97bafa3201d4e70eba2877e66e040` (22.07.2026). RocketPy publishes these flights with the
permission of each team, named in its example notebooks (`docs/examples/*_flight_sim.ipynb`).
The rocket data (geometry, masses, launch conditions) are in `configs/validation/<rocket>.yaml`,
with their source in the comments.

| Folder | Rocket, team, event | Propulsion | Apogee [m] | Log |
|---|---|---|---|---|
| `bella_lui` | Bella Lui, EPFL Rocket Team, Kaltbrunn 2020 | AeroTech K828FJ | 459 | altitude, speed |
| `cavour` | Cavour, PoliTo Rocket Team, EuRoC 2023 | Cesaroni L995 | 2789 | altitude, speed |
| `halcyon` | Halcyon, Aerospace Team Graz, EuRoC 2023 | N2O hybrid | 3450 | altitude |
| `andromeda` | Andromeda, Aristotle Space & Aeronautics Team, EuRoC 2022 | Cesaroni M2020 | 3443 | altitude, speed |
| `hedy` | Hedy, TU Wien Space Team, EuRoC 2025 | LOX / ethanol | 5232 | altitude, speed (CATS) |
| `defiance` | Defiance Mk. IV, University of Toronto, Launch Canada 2024 | N2O hybrid | 9308 | apogee only |

Files in each folder:

* `flight.csv`: `t_s` (from lift-off), `z_m` (height above the pad), `V_m_s` (speed, empty when the
  log has none), resampled at 20 Hz up to 5 s after apogee. The speeds are flight-computer
  estimates (barometer and accelerometer); near Mach 1 the barometric part is unreliable.
* `thrust.csv`: `t_s`, `F_N` from the release of the rocket. For Hedy the published curve starts at
  ignition and the rocket is held down 1.4 s; `thrust.csv` starts at release,
  `thrust_from_ignition.csv` is the reading of the RocketPy example (from ignition, cut at 8.05 s).
  The flight log suggests the real thrust lies between the two.

Regenerate with `python studies/flight_data_prep.py` (downloads the raw files into
`output/flight_data_raw/`, which is not in git).

"""Configuration schema, defaults, loading and validation.

Every design input of the tool is a field of one of the dataclasses below. Each field carries
its unit, valid range, allowed values and a one-line description in its metadata; the same
metadata produces ``python -m nosecone describe-config`` and the parameter tables of the user
guide, so the documentation cannot drift away from the code.

A configuration file (YAML) only needs the values that differ from the defaults. Relative paths
in the file are resolved from the folder that contains the file.
"""
from __future__ import annotations

import copy
import dataclasses as dc
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .constants import G0


# ------------------------------------------------------------------------- field helpers
@dataclass(frozen=True)
class Range:
    """Closed or open numeric interval used for validation and documentation."""

    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_open: bool = False
    hi_open: bool = False

    def contains(self, v: float) -> bool:
        """True if v lies in the interval."""
        if self.lo is not None and (v < self.lo or (self.lo_open and v == self.lo)):
            return False
        if self.hi is not None and (v > self.hi or (self.hi_open and v == self.hi)):
            return False
        return True

    def __str__(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"{'(' if self.lo_open else '['}{self.lo:g}, {self.hi:g}{')' if self.hi_open else ']'}"
        if self.lo is not None:
            return f"{'>' if self.lo_open else '>='} {self.lo:g}"
        if self.hi is not None:
            return f"{'<' if self.hi_open else '<='} {self.hi:g}"
        return "any"


POS = Range(0.0, None, lo_open=True)
NONNEG = Range(0.0, None)
FRACTION = Range(0.0, 1.0)


def param(default: Any, unit: str = "-", doc: str = "", valid: Optional[Range] = None,
          choices: Optional[tuple] = None, advanced: bool = False) -> Any:
    """Dataclass field with documentation metadata (unit, description, range, choices)."""
    meta = dict(unit=unit, doc=doc, valid=valid, choices=choices, advanced=advanced)
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: copy.deepcopy(d), metadata=meta)
    return field(default=default, metadata=meta)


# ------------------------------------------------------------------------------- sections
@dataclass
class ProjectConfig:
    """Identification of the study."""

    name: str = param("Weisshorn", doc="Vehicle or study name, used in plot titles and file names")
    output_dir: str = param("output", unit="path", doc="Folder for all outputs (created if missing)")
    plugins: list = param([], unit="path", doc="Python files imported before the run, e.g. new profile families (see the user guide)")


@dataclass
class VehicleConfig:
    """Launch vehicle data that do not depend on the nose."""

    diameter: float = param(0.180, "m", "External (body) diameter D; the nose base diameter", POS)
    length: float = param(5.4, "m", "Total vehicle length including the reference nose", POS)
    reference_nose_length: float = param(
        0.72, "m", "Nose length included in `length`; the body is length - reference_nose_length", POS)
    mass_wet: float = param(75.0, "kg", "Lift-off mass", POS)
    mass_dry: float = param(45.0, "kg", "Burn-out mass used by mass_model 'lvo'", POS)
    surface_roughness: float = param(20e-6, "m", "Equivalent sand roughness of the skin (painted CFRP ~ 20 um)", NONNEG)
    nozzle_exit_diameter: float = param(0.075, "m", "Nozzle exit diameter; removed from the base area while burning", NONNEG)


@dataclass
class EngineConfig:
    """Constant-thrust engine model."""

    thrust: float = param(4500.0, "N", "Thrust (constant over the burn)", POS)
    burn_time: float = param(9.1, "s", "Burn duration", POS)
    total_impulse: float = param(40960.0, "N s", "Total impulse (sets the propellant mass for mass_model 'isp')", POS)
    specific_impulse: float = param(200.0, "s", "Specific impulse (propellant mass = I_t / (I_sp g0))", POS)


@dataclass
class FinsConfig:
    """Trapezoidal fin set (drag build-up and Barrowman stability)."""

    count: int = param(4, "-", "Number of fins", Range(1, 8))
    root_chord: float = param(0.30, "m", "Root chord", POS)
    tip_chord: float = param(0.12, "m", "Tip chord", NONNEG)
    span: float = param(0.16, "m", "Semi-span (root to tip)", POS)
    sweep: float = param(0.18, "m", "Leading-edge sweep length (root LE to tip LE, axial)", NONNEG)
    thickness: float = param(0.005, "m", "Fin thickness", POS)
    edge_fraction: float = param(0.15, "-", "Length of each LE/TE wedge of the hexagonal section, as a fraction of the mean chord", Range(0.0, 0.5, lo_open=True))


@dataclass
class LaunchConfig:
    """Launch rail and site."""

    rail_length: float = param(12.0, "m", "Launch rail length", POS)
    rail_elevation: float = param(84.0, "deg", "Rail elevation above the horizontal", Range(0.0, 90.0, lo_open=True))
    site_altitude: float = param(0.0, "m", "Launch site altitude above sea level", Range(-500.0, 5000.0))


@dataclass
class AtmosphereConfig:
    """Standard-atmosphere model (troposphere with a ground-temperature offset)."""

    model: str = param("isa_troposphere", doc="Atmosphere model", choices=("isa_troposphere",))
    ground_temperature: float = param(288.15, "K", "Temperature at the launch site (288.15 K = ISA)", Range(200.0, 340.0))
    ground_pressure: float = param(101325.0, "Pa", "Pressure at the reference altitude of the model", POS)
    lapse_rate: float = param(0.0065, "K/m", "Temperature lapse rate in the troposphere", POS)
    tropopause_temperature: float = param(216.65, "K", "Temperature floor (tropopause / lower stratosphere)", POS)


@dataclass
class FlightConfig:
    """Trajectory model, calibration and envelope case."""

    mass_model: str = param("isp", doc="Nominal mass model: 'isp' (propellant = I_t/(I_sp g0)) or 'lvo' (wet -> dry)", choices=("isp", "lvo"))
    calibrate: bool = param(True, doc="Scale the non-nose drag by a factor k so that the baseline reaches target_apogee")
    target_apogee: float = param(9000.0, "m", "Apogee the baseline nose must reach when calibrate is true", POS)
    drag_factor: float = param(1.0, "-", "Factor k on the non-nose drag when calibrate is false", POS)
    calibration_bracket: list = param([0.3, 1.5], "-", "Search interval for k", advanced=True)
    calibration_tolerance: float = param(1e-4, "-", "Absolute tolerance on k", POS, advanced=True)
    t_max: float = param(120.0, "s", "Integration time limit (apogee normally ends the integration earlier)", POS, advanced=True)
    max_step: float = param(0.02, "s", "Maximum ODE time step", POS, advanced=True)
    rtol: float = param(1e-8, "-", "ODE relative tolerance", POS, advanced=True)
    atol: float = param(1e-8, "-", "ODE absolute tolerance", POS, advanced=True)
    output_step: float = param(0.05, "s", "Output sampling of the trade-study trajectories (sets M_max)", POS, advanced=True)
    detail_output_step: float = param(0.02, "s", "Output sampling of the trajectories used for heating and plots", POS, advanced=True)
    envelope_mass_model: str = param("lvo", doc="Mass model of the hot-day envelope trajectory", choices=("isp", "lvo"))
    envelope_ground_temperature: float = param(303.15, "K", "Ground temperature of the hot-day envelope", Range(200.0, 340.0))
    mass_sensitivity_step: float = param(1.0, "kg", "Mass step for the apogee sensitivity dz/dm", POS, advanced=True)
    cd_sensitivity_step: float = param(0.01, "-", "C_D step for the apogee sensitivity dz/dC_D", POS, advanced=True)


@dataclass
class NoseConfig:
    """Design space (the sweep) and the selected design."""

    profiles: list = param(["cone", "ogive", "parabolic", "power050", "power075", "lvhaack", "vonkarman"],
                           doc="Profiles compared in the sweep (built-in names or keys of custom_profiles)")
    fineness: list = param([3.0, 3.5, 4.0, 4.5, 5.0], "-", "Fineness ratios L/D compared in the sweep")
    baseline_profile: str = param("vonkarman", doc="Reference nose for apogee and static-margin changes")
    baseline_fineness: float = param(4.0, "-", "Fineness of the reference nose", POS)
    baseline_label: str = param("baseline", doc="How the reference length is called in table captions (e.g. 'LVS length')")
    design_profile: str = param("vonkarman", doc="Selected profile (CAD export, heating, final drawing)")
    design_fineness: float = param(4.5, "-", "Selected fineness L/D", POS)
    design_tip_radius_ratio: float = param(0.05, "-", "Selected spherical tip radius r_n / R (0 = sharp)", Range(0.0, 0.5))
    tip_radius_ratios: list = param([0.0], "-", "Tip radii r_n / R compared in the sweep (needs CFD data for each)")
    shoulder_length: float = param(0.060, "m", "Cylindrical shoulder behind the curved part (coupler bond)", NONNEG)
    custom_profiles: dict = param({}, doc="Extra profiles: name -> {family: ..., <parameter>: value, label: ...}")


@dataclass
class AerodynamicsConfig:
    """Nose wave-drag model built from the CFD database."""

    database: str = param("data/euler_results.csv", "path", "CSV of Euler wave-drag results (non-dimensional, reusable for any diameter)")
    user_database: str = param("", "path", "Optional second CSV for cases computed with compute_missing (empty: they are added to `database`)")
    studies: list = param(["main", "f45"], doc="Database studies used for the wave-drag model")
    grid: str = param("medium", doc="Grid level of the database rows used", choices=("coarse", "medium", "fine"))
    min_fineness: float = param(0.0, "-", "Database rows below this fineness are ignored", NONNEG, advanced=True)
    max_fineness: float = param(5.0, "-", "Database rows above this fineness are ignored", POS, advanced=True)
    machs: list = param([1.1, 1.2, 1.4, 1.6], "-", "Supersonic Mach numbers at which the wave drag is tabulated")
    transonic_onset: float = param(0.8, "-", "Mach number where the nose wave drag starts to rise", Range(0.5, 1.0))
    transonic_full: float = param(1.0, "-", "Mach number where the rise reaches the value at transonic_match", Range(0.8, 1.2))
    transonic_match: float = param(1.1, "-", "Lowest Euler Mach number; the model is flat between transonic_full and this", Range(1.0, 1.5))
    compute_missing: bool = param(False, doc="Run the Euler solver for profiles / fineness / tip radii missing from the database")
    missing_machs: list = param([1.6, 1.4, 1.2, 1.1], "-", "Mach sequence for cases computed on demand (highest first)")


@dataclass
class PayloadConfig:
    """Payload that must fit inside the nose bay."""

    enabled: bool = param(True, doc="Evaluate the payload fit and nose-bay length")
    name: str = param("3U", doc="Payload name used in labels")
    length: float = param(0.3405, "m", "Payload length", POS)
    width: float = param(0.100, "m", "Payload width (square section), used in drawings", POS)
    required_radius: float = param(0.077, "m", "Outer nose radius needed to house the payload section (half-diagonal + clearance + guide + shell)", POS)
    axial_clearance: float = param(0.010, "m", "Axial margin for the payload bulkhead", NONNEG)


@dataclass
class StructureConfig:
    """Nose structure mass model."""

    shell_areal_density: float = param(0.98, "kg/m^2", "Shell mass per wetted area (Firehorn as-built: 0.98)", POS)
    tip_mass: float = param(0.060, "kg", "Machined tip insert mass (same for all profiles)", NONNEG)
    shell_cg_fraction: float = param(0.35, "-", "Shell centre of gravity ahead of the shoulder, as a fraction of L", FRACTION, advanced=True)


@dataclass
class HeatingConfig:
    """Aerodynamic heating of the tip and the shell."""

    enabled: bool = param(True, doc="Run the heating analysis")
    trajectory_from: str = param("design", doc="Nose used to compute the heating trajectories", choices=("design", "baseline"))
    materials: dict = param({
        "CFRP": {"label": "CFRP (through-thickness)", "conductivity": 0.7, "density": 1550.0, "specific_heat": 900.0},
        "Al6082": {"label": "Al 6082", "short": "Al", "conductivity": 180.0, "density": 2700.0, "specific_heat": 900.0},
        "PETCF": {"label": "PET-CF", "conductivity": 0.3, "density": 1300.0, "specific_heat": 1200.0},
    }, doc="Material properties: name -> {label, short (optional), conductivity [W/(m K)], density [kg/m^3], specific_heat [J/(kg K)]}")
    shell_material: str = param("CFRP", doc="Shell material (key of materials)")
    shell_thickness: float = param(0.0012, "m", "Shell thickness", POS)
    shell_stations: list = param(["0.05 m", "0.5 L", "1.0 L"], doc="Shell stations: 'x m' from the tip or 'f L' as a fraction of L")
    tip_material: str = param("Al6082", doc="Tip insert material (lumped model)")
    tip_length: float = param(0.08, "m", "Axial length of the tip insert from the apex", POS)
    printed_tip_material: str = param("PETCF", doc="Material of the comparison printed tip (1D wall at the stagnation point)")
    printed_tip_thickness: float = param(0.004, "m", "Wall thickness of the printed tip", POS)
    wall_nodes: int = param(40, "-", "Nodes through the wall thickness", Range(5, 400), advanced=True)
    bond_limit: float = param(120.0, "degC", "Adhesive temperature limit drawn on the shell plot", None)
    bond_label: str = param("DP490 bond limit", doc="Label of the adhesive limit")
    printed_tip_hdt: list = param([70.0, 80.0], "degC", "Heat-deflection temperature range of the printed tip polymer")


@dataclass
class BluntnessConfig:
    """Effect of a spherical tip, from the CFD database."""

    enabled: bool = param(True, doc="Evaluate the tip-bluntness study")
    study: str = param("blunt", doc="Database study with blunted noses")
    profile: str = param("vonkarman", doc="Profile of the bluntness study")
    fineness: float = param(4.0, "-", "Fineness of the bluntness study", POS)
    report_ratios: list = param([0.05, 0.1], "-", "Tip radii r_n/R whose drag and apogee cost are reported")
    report_mach: float = param(1.4, "-", "Mach number at which the drag cost of the tip is reported", Range(1.0, 5.0))
    radius_range: list = param([0.002, 0.020], "m", "Tip radius range of the heat-transfer curve")


@dataclass
class RobustnessConfig:
    """Ranking of the profiles under perturbed modelling assumptions."""

    enabled: bool = param(True, doc="Run the robustness study")
    fineness: float = param(4.5, "-", "Fineness at which all profiles are compared", POS)
    reference_profile: str = param("vonkarman", doc="Apogees are shown relative to this profile")
    band: float = param(30.0, "m", "Half-width of the 'equivalent' band in the plot", NONNEG)
    scenarios: list = param([
        {"name": "nominal"},
        {"name": "drag rise from M = 0.75", "transonic_onset": 0.75},
        {"name": "drag rise from M = 0.85", "transonic_onset": 0.85},
        {"name": "transonic peak +20 %", "transonic_factor": 1.2},
        {"name": "transonic peak -20 %", "transonic_factor": 0.8},
        {"name": "rest of vehicle drag +20 %", "drag_factor_scale": 1.2},
        {"name": "LVO masses, 30 C day", "mass_model": "lvo", "ground_temperature": 303.15},
    ], doc="Scenarios: name + any of transonic_onset, transonic_factor, drag_factor_scale, mass_model, ground_temperature")


@dataclass
class TradeMatrixConfig:
    """Weighted decision matrix (scores 1-5)."""

    enabled: bool = param(True, doc="Build the weighted trade matrix")
    fineness: float = param(4.5, "-", "Fineness at which the matrix is built", POS)
    weights: dict = param({"performance": 4, "integration": 2, "manufacture": 2, "stability": 1},
                          doc="Relative criterion weights, normalised to add up to 1 (criteria: performance, "
                              "integration, manufacture, stability)")
    manufacture: dict = param({"cone": 3, "ogive": 5, "parabolic": 5, "power050": 3, "power075": 2,
                               "lvhaack": 5, "vonkarman": 5}, doc="Judgement score per profile (1-5)")


@dataclass
class ValidationConfig:
    """CFD validation plots (only drawn if the studies exist in the database)."""

    cone_study: str = param("cones", doc="Database study of sharp cones (compared with Taylor-Maccoll)")
    cone_main_fineness: float = param(4.0, "-", "Cone of the main study added to the validation plot", POS)
    grid_study: str = param("grid", doc="Database study of the grid-convergence runs")
    grid_profile: str = param("vonkarman", doc="Profile of the grid study")
    grid_fineness: float = param(4.0, "-", "Fineness of the grid study", POS)
    grid_machs: list = param([1.4, 1.2], "-", "Mach numbers of the grid study")


@dataclass
class CFDConfig:
    """Axisymmetric Euler solver: grids, numerics and studies."""

    cp_dir: str = param("data/cfd/cp", "path", "Folder of the wall-pressure files (one .npz per case)")
    fields_dir: str = param("data/cfd/fields", "path", "Folder of the saved flow fields")
    log: str = param("data/euler_sweep_log.csv", "path", "Run log appended by the solver")
    workers: int = param(4, "-", "Parallel processes", Range(1, 64))
    cfl: float = param(0.85, "-", "CFL number of the local time stepping", Range(0.05, 1.0))
    fallback_cfl: float = param(0.5, "-", "CFL of the restart from free stream if a run diverges", Range(0.05, 1.0), advanced=True)
    max_iter: int = param(6000, "-", "Iteration limit per Mach number", Range(100, None))
    fine_iter_factor: int = param(2, "-", "Iteration-limit multiplier on the fine grid", Range(1, 10), advanced=True)
    tolerance: float = param(2e-5, "-", "Convergence: relative change of C_D,w over `check` iterations", POS)
    check: int = param(250, "-", "Iterations between convergence checks", Range(10, None), advanced=True)
    first_order_iters: int = param(300, "-", "First-order iterations at the start of the first Mach number", NONNEG, advanced=True)
    grids: dict = param({"coarse": [60, 20, 40, 4e-3], "medium": [120, 40, 80, 2e-3], "fine": [240, 80, 160, 1e-3]},
                        doc="Grid levels: name -> [cells along nose, cells along cylinder, cells normal, first-cell height / R]")
    cylinder_length: float = param(5.0, "R", "Cylinder behind the nose in the CFD domain", POS, advanced=True)
    upstream_distance: float = param(1.0, "R", "Distance of the outer boundary ahead of the tip (scaled by 1 + 0.3 L/D)", POS, advanced=True)
    outer_angle: float = param(72.0, "deg", "Asymptotic angle of the outer boundary (hyperbola)", Range(30.0, 89.0), advanced=True)
    save_fields: list = param([1.4, 1.2], "-", "Mach numbers at which flow fields are saved (studies with save_fields: true)", advanced=True)
    studies: list = param([], doc="Euler studies: list of {name, profiles, fineness, tip_radius_ratios, grid, machs, save_fields}")


@dataclass
class CADConfig:
    """Geometry export."""

    profile_csv: bool = param(True, doc="Write the profile coordinates of the design (x from the tip, r) in mm")
    profile_points: int = param(200, "-", "Points of the profile CSV", Range(10, 100000))
    profile_clustering: float = param(1.6, "-", "Clustering exponent towards the tip of the CSV points (1 = uniform)", Range(1.0, 4.0), advanced=True)
    stl: bool = param(True, doc="Write watertight STL solids")
    stl_facets: int = param(128, "-", "Azimuthal facets of the STL", Range(8, 4096))
    stl_points: int = param(220, "-", "Profile points of the candidate STLs", Range(10, 100000), advanced=True)
    stl_points_design: int = param(400, "-", "Profile points of the design STL", Range(10, 100000), advanced=True)
    step: bool = param(True, doc="Write a STEP solid of the design (needs the gmsh package)")
    step_points: int = param(200, "-", "Profile points through which the STEP B-spline is fitted", Range(10, 5000), advanced=True)
    candidates: bool = param(True, doc="Also export every profile of the sweep at the design fineness (STL)")


@dataclass
class OutputConfig:
    """What to produce."""

    formats: list = param(["pdf", "png"], doc="Figure file formats", choices=("pdf", "png", "svg"))
    png_dpi: int = param(200, "dpi", "Resolution of PNG figures", Range(50, 1200), advanced=True)
    plots: list = param(["all"], doc="Plots to draw: 'all' or a list of names (see the user guide)")
    latex: bool = param(False, doc="Also write LaTeX tables and macros (design-report style)")
    text_renderer: str = param("auto", doc="Plot text: 'latex' (needs a LaTeX installation), 'mathtext' (built in) or 'auto'",
                               choices=("auto", "latex", "mathtext"))
    plot_options: dict = param({}, doc="Per-plot overrides, e.g. {trajectory: {xlim: [0, 40]}}")


@dataclass
class Config:
    """Complete configuration."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    fins: FinsConfig = field(default_factory=FinsConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    nose: NoseConfig = field(default_factory=NoseConfig)
    aerodynamics: AerodynamicsConfig = field(default_factory=AerodynamicsConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    heating: HeatingConfig = field(default_factory=HeatingConfig)
    bluntness: BluntnessConfig = field(default_factory=BluntnessConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    trade_matrix: TradeMatrixConfig = field(default_factory=TradeMatrixConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    cfd: CFDConfig = field(default_factory=CFDConfig)
    cad: CADConfig = field(default_factory=CADConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    base_dir: Path = field(default=Path("."), metadata=dict(internal=True))

    # ------------------------------------------------------------------ conveniences
    @property
    def R(self) -> float:
        """Base radius [m]."""
        return self.vehicle.diameter / 2

    def path(self, p: str) -> Path:
        """Resolve a path of the configuration relative to the configuration file."""
        q = Path(p)
        return q if q.is_absolute() else (self.base_dir / q).resolve()

    @property
    def out(self) -> Path:
        """Output folder (resolved)."""
        return self.path(self.project.output_dir)


SECTIONS: dict[str, type] = {f.name: f.type if isinstance(f.type, type) else globals()[f.type]
                             for f in dc.fields(Config) if f.name != "base_dir"}


# ------------------------------------------------------------------------------ loading
class ConfigError(ValueError):
    """Raised for an invalid configuration, with a message pointing at the offending key."""


_ADD_TO_DEFAULTS = {"materials", "grids"}
"""Mapping parameters whose entries are added to the default entries instead of replacing them."""


def _build(cls: type, data: dict, where: str) -> Any:
    names = {f.name: f for f in dc.fields(cls)}
    unknown = [k for k in data if k not in names]
    if unknown:
        k = unknown[0]
        hint = difflib.get_close_matches(k, names, n=1)
        raise ConfigError(f"unknown parameter '{where}.{k}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
    obj = cls()
    for k, v in data.items():
        default = getattr(obj, k)
        if isinstance(default, bool):
            if not isinstance(v, bool):
                raise ConfigError(f"'{where}.{k}' must be true or false, got {v!r}")
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
                raise ConfigError(f"'{where}.{k}' must be an integer, got {v!r}")
            v = int(v)
        elif isinstance(default, float):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"'{where}.{k}' must be a number, got {v!r}")
            v = float(v)
        elif isinstance(default, str) and not isinstance(v, str):
            raise ConfigError(f"'{where}.{k}' must be a text value, got {v!r}")
        elif isinstance(default, list) and not isinstance(v, list):
            raise ConfigError(f"'{where}.{k}' must be a list, got {v!r}")
        elif isinstance(default, dict) and not isinstance(v, dict):
            raise ConfigError(f"'{where}.{k}' must be a mapping, got {v!r}")
        if isinstance(default, dict) and k in _ADD_TO_DEFAULTS:
            v = {**default, **v}          # e.g. a new material is added to the default ones
        setattr(obj, k, v)
    return obj


def from_dict(data: dict, base_dir: Path | str = ".") -> Config:
    """Build and validate a :class:`Config` from a (partial) nested dictionary."""
    data = data or {}
    unknown = [k for k in data if k not in SECTIONS]
    if unknown:
        hint = difflib.get_close_matches(unknown[0], SECTIONS, n=1)
        raise ConfigError(f"unknown section '{unknown[0]}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
    kw = {}
    for name, cls in SECTIONS.items():
        sec = data.get(name) or {}
        if not isinstance(sec, dict):
            raise ConfigError(f"section '{name}' must be a mapping")
        kw[name] = _build(cls, sec, name)
    cfg = Config(**kw, base_dir=Path(base_dir).resolve())
    validate(cfg)
    return cfg


PATH_FIELDS = [("project", "output_dir"), ("aerodynamics", "database"), ("aerodynamics", "user_database"),
               ("cfd", "cp_dir"), ("cfd", "fields_dir"), ("cfd", "log")]


def _read_yaml(path: Path, _seen: tuple = ()) -> dict:
    """YAML data of a file, merged onto its `base:` file if it names one (paths made absolute)."""
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    if path.resolve() in _seen:
        raise ConfigError(f"circular 'base:' reference involving {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the file must contain sections (a YAML mapping)")
    for sec, key in PATH_FIELDS:                  # relative paths are relative to the file that sets them
        v = (data.get(sec) or {}).get(key)
        if isinstance(v, str) and v and not Path(v).is_absolute():
            data[sec][key] = str((path.parent / v).resolve())
    plug = (data.get("project") or {}).get("plugins")
    if isinstance(plug, list):
        data["project"]["plugins"] = [str((path.parent / q).resolve()) if not Path(q).is_absolute() else q for q in plug]
    base = data.pop("base", None)
    if base is None:
        return data
    merged = _read_yaml((path.parent / base), _seen + (path.resolve(),))
    return _merge(merged, data)


def _merge(a: dict, b: dict) -> dict:
    """Deep merge: mappings are merged key by key, everything else (lists included) is replaced."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in _REPLACE else v
    return out


_REPLACE = {"weights", "manufacture", "grids"}
"""Mapping parameters that are replaced as a whole (not merged) when a file overrides its base."""


def load(path: str | Path) -> Config:
    """Read a YAML configuration file (missing values take their defaults; `base:` files are inherited)."""
    path = Path(path)
    data = _read_yaml(path)
    plugins = (data.get("project") or {}).get("plugins") or []
    if plugins:
        from .plugins import load_plugins
        try:
            load_plugins(plugins)
        except Exception as e:  # report plugin errors as configuration errors
            raise ConfigError(f"project.plugins: {type(e).__name__}: {e}") from e
    return from_dict(data, base_dir=path.parent)


def to_dict(cfg: Config) -> dict:
    """Configuration as a nested dictionary (for saving next to the outputs)."""
    return {name: dc.asdict(getattr(cfg, name)) for name in SECTIONS}


def save(cfg: Config, path: str | Path) -> None:
    """Write the full configuration (with defaults filled in) to a YAML file."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, allow_unicode=True)


# ------------------------------------------------------------------------------ validation
def _check_ranges(sec_name: str, sec: Any) -> None:
    for f in dc.fields(sec):
        v = getattr(sec, f.name)
        rng: Optional[Range] = f.metadata.get("valid")
        choices = f.metadata.get("choices")
        if rng is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and not rng.contains(v):
            raise ConfigError(f"'{sec_name}.{f.name}' = {v} is outside the valid range {rng}")
        if choices is not None:
            vals = v if isinstance(v, list) else [v]
            bad = [x for x in vals if x not in choices]
            if bad:
                raise ConfigError(f"'{sec_name}.{f.name}' = {bad[0]!r} is not one of {list(choices)}")


def validate(cfg: Config) -> None:
    """Check ranges, choices and cross-references; raise :class:`ConfigError` on the first problem."""
    from .geometry import profile_names, make_profiles  # local import: geometry needs no config

    for name in SECTIONS:
        _check_ranges(name, getattr(cfg, name))
    n = cfg.nose
    try:
        make_profiles(n.custom_profiles)
    except (ValueError, TypeError, KeyError) as e:
        raise ConfigError(f"nose.custom_profiles: {e}") from e
    known = set(profile_names(n.custom_profiles))
    for key, val in [("nose.profiles", n.profiles), ("nose.baseline_profile", [n.baseline_profile]),
                     ("nose.design_profile", [n.design_profile]),
                     ("robustness.reference_profile", [cfg.robustness.reference_profile]),
                     ("bluntness.profile", [cfg.bluntness.profile])]:
        for p in val:
            if p not in known:
                hint = difflib.get_close_matches(p, known, n=1)
                raise ConfigError(f"{key}: unknown profile '{p}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
    if not n.profiles:
        raise ConfigError("nose.profiles must list at least one profile")
    if not n.fineness or any((not isinstance(f, (int, float))) or f <= 0 for f in n.fineness):
        raise ConfigError("nose.fineness must be a non-empty list of positive numbers")
    if any((not isinstance(r, (int, float))) or not 0 <= r <= 0.5 for r in n.tip_radius_ratios):
        raise ConfigError("nose.tip_radius_ratios must be numbers in [0, 0.5]")
    v = cfg.vehicle
    if v.reference_nose_length >= v.length:
        raise ConfigError("vehicle.reference_nose_length must be shorter than vehicle.length")
    if v.mass_dry >= v.mass_wet:
        raise ConfigError("vehicle.mass_dry must be smaller than vehicle.mass_wet")
    e = cfg.engine
    m_prop = e.total_impulse / (e.specific_impulse * G0)
    if m_prop >= v.mass_wet:
        raise ConfigError(f"engine: propellant mass I_t/(I_sp g0) = {m_prop:.1f} kg exceeds vehicle.mass_wet")
    a = cfg.aerodynamics
    if sorted(a.machs) != list(a.machs) or len(a.machs) < 2:
        raise ConfigError("aerodynamics.machs must be at least two increasing Mach numbers")
    if not a.transonic_onset < a.transonic_full <= a.transonic_match:
        raise ConfigError("need aerodynamics.transonic_onset < transonic_full <= transonic_match")
    if len(cfg.flight.calibration_bracket) != 2 or not 0 < cfg.flight.calibration_bracket[0] < cfg.flight.calibration_bracket[1]:
        raise ConfigError("flight.calibration_bracket must be [k_min, k_max] with 0 < k_min < k_max")
    h = cfg.heating
    for key in ("shell_material", "tip_material", "printed_tip_material"):
        if getattr(h, key) not in h.materials:
            raise ConfigError(f"heating.{key} = '{getattr(h, key)}' is not defined in heating.materials")
    for mname, m in h.materials.items():
        for prop in ("conductivity", "density", "specific_heat"):
            if not isinstance(m.get(prop), (int, float)) or m[prop] <= 0:
                raise ConfigError(f"heating.materials.{mname}.{prop} must be a positive number")
    for s in h.shell_stations:
        parse_station(s, 1.0)
    for g, spec in cfg.cfd.grids.items():
        if not (isinstance(spec, list) and len(spec) == 4):
            raise ConfigError(f"cfd.grids.{g} must be [n_nose, n_cylinder, n_normal, first_cell_height]")
    if a.grid not in cfg.cfd.grids:
        raise ConfigError(f"aerodynamics.grid '{a.grid}' is not defined in cfd.grids")
    for sc in cfg.robustness.scenarios:
        extra = set(sc) - {"name", "transonic_onset", "transonic_factor", "drag_factor_scale", "mass_model",
                           "ground_temperature"}
        if "name" not in sc or extra:
            raise ConfigError(f"robustness.scenarios: bad entry {sc} (needs 'name'; unknown keys {sorted(extra)})")
    w = cfg.trade_matrix.weights
    if set(w) - {"performance", "integration", "manufacture", "stability"}:
        raise ConfigError("trade_matrix.weights: allowed keys are performance, integration, manufacture, stability")
    if any(not isinstance(v, (int, float)) or v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ConfigError("trade_matrix.weights: weights must be non-negative numbers with a positive sum")
    for st in cfg.cfd.studies:
        missing = {"name", "profiles", "fineness", "machs"} - set(st)
        if missing:
            raise ConfigError(f"cfd.studies: study {st.get('name', '?')} misses {sorted(missing)}")


def parse_station(s: Any, L: float) -> float:
    """Shell station as a distance from the tip [m]: number (m), 'x m' or 'f L'."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if isinstance(s, str):
        t = s.strip()
        try:
            if t.endswith("L"):
                return float(t[:-1]) * L
            if t.endswith("m"):
                return float(t[:-1])
            return float(t)
        except ValueError:
            pass
    raise ConfigError(f"heating.shell_stations: cannot read station {s!r} (use e.g. '0.05 m' or '0.5 L')")


# --------------------------------------------------------------------------- documentation
def describe(include_advanced: bool = True) -> list[dict]:
    """All parameters with section, name, default, unit, valid range and description."""
    rows = []
    for name, cls in SECTIONS.items():
        for f in dc.fields(cls):
            m = f.metadata
            if not include_advanced and m.get("advanced"):
                continue
            default = f.default if f.default is not dc.MISSING else f.default_factory()  # type: ignore[misc]
            valid = m.get("valid")
            choices = m.get("choices")
            rows.append(dict(section=name, name=f.name, default=default, unit=m.get("unit", "-"),
                             valid=(", ".join(map(str, choices)) if choices else (str(valid) if valid else "")),
                             doc=m.get("doc", ""), advanced=bool(m.get("advanced"))))
    return rows

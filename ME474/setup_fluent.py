"""Fluent 2024 R2 setup of the ME-474 nose-cone runs: 2D axisymmetric, steady RANS k-omega SST.

Copied from the Rocket Team study (CFD/study720/setup_fluent.py), with these changes for the course:
* the pressure-outlet backflow total temperature is the free-stream T0, not the static T_inf;
* mass and heat flux reports on every boundary, logged each iteration to <run>_flux.out, for the
  week 8 balance check (net imbalance within 0.2 %);
* after convergence: the axis export (the stagnation streamline, inlet to tip) and a transcript
  with the wall forces (pressure / viscous split) and the mesh check and quality;
* every output of a run goes to runs/<run>/; the 300 K wall stage is dropped (adiabatic walls only).

Normally driven by run_queue.py. A stage can also be run alone on the Fluent session whose
server_info*.txt in this folder is the newest:
    python setup_fluent.py general
"""
import sys
from pathlib import Path

import ansys.fluent.core as pf

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

R_AIR = 8314.462618 / 28.966          # Fluent's default air [J/(kg K)]
GAMMA = 1006.43 / (1006.43 - R_AIR)   # 1.399
A_REF = 0.025447                      # pi * 0.09^2 [m2]
WALLS = ["wall_nose", "wall_coupler", "wall_body"]
FLOW_BOUNDARIES = ["inlet", "farfield", "outlet"]
P_INF, M_INF, T_INF = 77684.7, 1.46548, 273.947  # burnout, fluent_points.csv


def run_dir(run):
    d = RUNS / run
    (d / "autosave").mkdir(parents=True, exist_ok=True)
    return d


def server_file():
    """The newest server_info*.txt in this folder: Fluent writes server_info-<pid>.txt on server/start-server."""
    files = sorted(HERE.glob("server_info*.txt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f'No server_info*.txt in {HERE}. In the Fluent console: (chdir "{HERE.as_posix()}"), '
                         "define/beta-feature-access (yes), then server/start-server.")
    return files[-1]


def connect():
    si = server_file()
    print("connecting to the Fluent session of", si.name, flush=True)
    address, password = si.read_text().split()
    ip, _, port = address.rpartition(":")
    try:
        return pf.connect_to_fluent(ip="127.0.0.1", port=int(port), password=password, cleanup_on_exit=False)
    except Exception:
        return pf.connect_to_fluent(ip=ip, port=int(port), password=password, allow_remote_host=True,
                                    insecure_mode=True, cleanup_on_exit=False)


def general(s):
    setup = s.settings.setup
    solver = setup.general.solver
    solver.type = "density-based-implicit"
    solver.two_dim_space = "axisymmetric"
    solver.time = "steady"
    setup.general.operating_conditions.operating_pressure = 0.0
    setup.models.energy.enabled = True
    setup.models.viscous.model = "k-omega"
    setup.models.viscous.k_omega_model = "sst"
    air = setup.materials.fluid["air"]
    air.density.option = "ideal-gas"
    air.viscosity.option = "sutherland"
    print("solver:", solver())
    print("operating pressure:", setup.general.operating_conditions.operating_pressure())
    print("viscous:", setup.models.viscous.model(), setup.models.viscous.k_omega_model())
    print("air:", {k: air()[k] for k in ("density", "viscosity", "specific_heat", "thermal_conductivity")})
    bc = setup.boundary_conditions
    bc.set_zone_type(zone_list=["inlet", "farfield"], new_type="pressure-far-field")
    print("\npressure far-field inlet:", bc.pressure_far_field["inlet"]())
    print("\npressure outlet:", bc.pressure_outlet["outlet"]())
    print("\nwall_nose:", bc.wall["wall_nose"]())


def flux_reports(s, run):
    """Mass and heat flux through each boundary and their net sum, every iteration, to <run>_flux.out."""
    flux = s.settings.solution.report_definitions.flux
    names = []
    for kind, tag, zones in (("flux-massflow", "m", FLOW_BOUNDARIES),
                             ("flux-heattransfer", "q", FLOW_BOUNDARIES + WALLS)):
        for z in zones + ["net"]:
            name = f"{tag}-{z}"
            try:
                flux[name] = {"report_type": kind}
            except Exception:
                print(f"report type {kind} refused; allowed:", flux[name].report_type.allowed_values())
                raise
            key = "boundaries" if "boundaries" in flux[name]() else "zone_names"  # name changed across versions
            flux[name] = {key: zones if z == "net" else [z]}
            names.append(name)
    s.settings.solution.monitor.report_files["flux-file"] = {
        "report_defs": names, "file_name": (run_dir(run) / f"{run}_flux.out").as_posix()}
    print("flux reports:", names)


def bcs(s, run, p_inf=P_INF, m_inf=M_INF, t_inf=T_INF):
    t0 = t_inf * (1 + 0.5 * (GAMMA - 1) * m_inf ** 2)
    setup = s.settings.setup
    bc = setup.boundary_conditions
    turb = {"turbulence_specification": "Intensity and Viscosity Ratio"}
    for z in ("inlet", "farfield"):
        bc.pressure_far_field[z] = {
            "momentum": {"gauge_pressure": {"value": p_inf}, "mach_number": {"value": m_inf},
                         "flow_direction": [{"value": 1}, {"value": 0}]},
            "thermal": {"temperature": {"value": t_inf}},
            "turbulence": {**turb, "turbulent_intensity": 0.01, "turbulent_viscosity_ratio": 10},
        }
    bc.pressure_outlet["outlet"] = {  # air re-entering through the outlet comes in at the free-stream T0
        "momentum": {"gauge_pressure": {"value": p_inf}, "backflow_dir_spec_method": "Normal to Boundary"},
        "thermal": {"backflow_total_temperature": {"value": t0}},
        "turbulence": {**turb, "backflow_turbulent_intensity": 0.01, "backflow_turbulent_viscosity_ratio": 10},
    }
    rv = setup.reference_values
    rv.compute(from_zone_type="pressure-far-field", from_zone_name="inlet")
    rv.area = A_REF
    rv.length = 0.18

    d = run_dir(run)
    sol = s.settings.solution
    drag = sol.report_definitions.drag
    for name, zones in (("cd-total", WALLS), ("cd-nose", ["wall_nose"])):
        drag[name] = {"zones": zones, "force_vector": [1, 0], "report_output_type": "Drag Coefficient"}
    sol.monitor.report_files["cd-file"] = {"report_defs": ["cd-total", "cd-nose"],
                                           "file_name": (d / f"{run}_cd.out").as_posix()}
    sol.monitor.report_plots["cd-plot"] = {"report_defs": ["cd-total", "cd-nose"]}
    sol.monitor.residual.options.criterion_type = "none"  # stop on the C_D criterion, not residuals
    flux_reports(s, run)
    s.settings.file.auto_save.data_frequency = 500
    s.settings.file.auto_save.root_name = (d / "autosave" / run).as_posix()

    for z in ("inlet", "farfield"):
        print(z, bc.pressure_far_field[z]())
    print("outlet", bc.pressure_outlet["outlet"](), f"(T0 = {t0:.3f} K)")
    print("reference values", rv())
    for name in ("cd-total", "cd-nose"):
        dd = drag[name]()
        print(name, {k: dd[k] for k in ("zones", "force_vector", "report_output_type")})
    print("report files", sol.monitor.report_files())
    print("residual criterion", sol.monitor.residual.options.criterion_type())
    print("autosave", s.settings.file.auto_save())


def methods(s, order="first-order-upwind", courant=1.0):
    """Density-based: Roe-FDS, least-squares gradients, first order and CFL 1 to start."""
    sol = s.settings.solution
    sol.methods.flux_type.dbns_cases.flux_type = "Roe-FDS"
    sol.methods.gradient_scheme = "least-square-cell-based"
    for eq in ("amg-c", "k", "omega"):
        sol.methods.discretization_scheme[eq] = order
    sol.controls.courant_number = courant
    m = sol.methods()
    print("methods:", m["flux_type"], m["gradient_scheme"], m["discretization_scheme"])
    print("courant:", sol.controls.courant_number())


def init(s, run):
    sol = s.settings.solution
    sol.initialization.initialization_type = "hybrid"
    sol.initialization.hybrid_initialize()
    s.settings.file.write(file_type="case-data", file_name=(run_dir(run) / f"{run}_init").as_posix())
    print("initialised and saved", f"{run}_init.cas.h5 / .dat.h5")


def pressure_based(s):
    """Study A's M 0.95 recipe: pressure-based Coupled with the pseudo-time method, viscous heating on."""
    setup = s.settings.setup
    setup.general.solver.type = "pressure-based"
    vo = setup.models.viscous.options
    vo.viscous_heating = True
    vo.compressibility_effects = False
    m = s.settings.solution.methods
    m.p_v_coupling.flow_scheme = "Coupled"
    ct = m.pseudo_time_method.formulation.coupled_solver
    ct.set_state(next(v for v in ct.allowed_values() if "global" in v))
    print("solver:", setup.general.solver(), "| viscous options:",
          {k: vo()[k] for k in ("viscous_heating", "compressibility_effects")})
    print("p-v coupling:", m.p_v_coupling.flow_scheme(), "| pseudo time:", ct())


def pb_order(s, order):
    """order 'first' or 'second': pressure second order throughout, the other equations upwind."""
    ds = s.settings.solution.methods.discretization_scheme
    for eq in ds():
        ds[eq] = "second-order" if eq == "pressure" else f"{order}-order-upwind"
    print("discretization:", ds())


def schedule_pb(s, run, first_iters=200, second_iters=1000):
    """Pressure-based: 200 iterations first order, then second order (study A at M 0.95)."""
    it = s.settings.solution.run_calculation.iterate
    it(iter_count=first_iters)
    pb_order(s, "second")
    it(iter_count=second_iters)
    s.settings.file.write(file_type="case-data", file_name=(run_dir(run) / run).as_posix())
    print("schedule done, saved", run)


def schedule(s, run, second_order_iters=3000):
    """Density-based: 500 first order at CFL 1, 500 at CFL 5, then second order at CFL 5."""
    it = s.settings.solution.run_calculation.iterate
    it(iter_count=500)
    methods(s, courant=5.0)
    it(iter_count=500)
    methods(s, order="second-order-upwind", courant=5.0)
    it(iter_count=second_order_iters)
    s.settings.file.write(file_type="case-data", file_name=(run_dir(run) / run).as_posix())
    print("schedule done, saved", run)


def export_wall(s, run):
    s.tui.file.export.ascii((run_dir(run) / f"{run}_wall.csv").as_posix(), "wall_nose", "wall_coupler", "()",
                            "yes", "pressure", "pressure-coefficient", "temperature", "heat-flux", "y-plus",
                            "skin-friction-coef", "wall-shear", "()", "yes")


def export_axis(s, run):
    """The axis runs from the inlet to the tip: it is the stagnation streamline (total-pressure check)."""
    s.tui.file.export.ascii((run_dir(run) / f"{run}_axis.csv").as_posix(), "axis", "()",
                            "yes", "pressure", "total-pressure", "mach-number", "temperature",
                            "total-temperature", "()", "yes")


def reports(s, run):
    """Wall forces (pressure and viscous parts), mesh check and mesh quality, to <run>_report.txt."""
    path = run_dir(run) / f"{run}_report.txt"
    if path.exists():
        path.unlink()
    s.tui.file.start_transcript(path.as_posix())
    try:
        s.tui.report.forces.wall_forces("yes", "1", "0", "no")
        s.tui.mesh.check()
        s.tui.mesh.quality()
    finally:
        s.tui.file.stop_transcript()


if __name__ == "__main__":
    s = connect()
    stage = sys.argv[1]
    if stage in ("bcs", "init", "export_wall", "export_axis", "reports"):
        {"bcs": bcs, "init": init, "export_wall": export_wall, "export_axis": export_axis,
         "reports": reports}[stage](s, sys.argv[2])
    else:
        {"general": general, "methods": methods, "pressure_based": pressure_based}[stage](s)

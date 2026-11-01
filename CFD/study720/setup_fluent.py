"""Fluent 2024 R2 setup of the 720 mm runs (run sheet step 3), on the session behind server_info-A.txt.

Usage: python setup_fluent.py <stage>
  general  solver, models, material, zone types; prints the boundary-condition fields
"""
import sys
from pathlib import Path

import ansys.fluent.core as pf

HERE = Path(__file__).resolve().parent
SI = HERE / "server_info-A.txt"


def connect():
    address, password = SI.read_text().split()
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


P_INF, M_INF, T_INF = 77685.0, 1.46548, 273.95  # burnout point, t = 9.10 s, z = 2185 m
A_REF = 0.025447  # pi * 0.09^2
WALLS = ["wall_nose", "wall_coupler", "wall_body"]


def bcs(s, run="A_adiab", p_inf=P_INF, m_inf=M_INF, t_inf=T_INF):
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
    bc.pressure_outlet["outlet"] = {
        "momentum": {"gauge_pressure": {"value": p_inf}, "backflow_dir_spec_method": "Normal to Boundary"},
        "thermal": {"backflow_total_temperature": {"value": t_inf}},
        "turbulence": {**turb, "backflow_turbulent_intensity": 0.01, "backflow_turbulent_viscosity_ratio": 10},
    }
    rv = setup.reference_values
    rv.compute(from_zone_type="pressure-far-field", from_zone_name="inlet")
    rv.area = A_REF
    rv.length = 0.18

    sol = s.settings.solution
    drag = sol.report_definitions.drag
    for name, zones in (("cd-total", WALLS), ("cd-nose", ["wall_nose"])):
        drag[name] = {"zones": zones, "force_vector": [1, 0], "report_output_type": "Drag Coefficient"}
    sol.monitor.report_files["cd-file"] = {"report_defs": ["cd-total", "cd-nose"],
                                           "file_name": (HERE / f"{run}_cd.out").as_posix()}
    sol.monitor.report_plots["cd-plot"] = {"report_defs": ["cd-total", "cd-nose"]}
    sol.monitor.residual.options.criterion_type = "none"  # stop on the C_D criterion, not residuals
    s.settings.file.auto_save.data_frequency = 500
    s.settings.file.auto_save.root_name = (HERE / "autosave" / run).as_posix()

    for z in ("inlet", "farfield"):
        print(z, bc.pressure_far_field[z]())
    print("outlet", bc.pressure_outlet["outlet"]())
    print("reference values", rv())
    for name in ("cd-total", "cd-nose"):
        d = drag[name]()
        print(name, {k: d[k] for k in ("zones", "force_vector", "report_output_type")})
    print("report file", sol.monitor.report_files["cd-file"]())
    print("residual criterion", sol.monitor.residual.options.criterion_type())
    print("autosave", s.settings.file.auto_save())


def methods(s, order="first-order-upwind", courant=1.0):
    """Run sheet step 4: Roe-FDS, least-squares gradients, first order and CFL 1 to start."""
    sol = s.settings.solution
    sol.methods.flux_type.dbns_cases.flux_type = "Roe-FDS"
    sol.methods.gradient_scheme = "least-square-cell-based"
    for eq in ("amg-c", "k", "omega"):
        sol.methods.discretization_scheme[eq] = order
    sol.controls.courant_number = courant
    m = sol.methods()
    print("methods:", m["flux_type"], m["gradient_scheme"], m["discretization_scheme"])
    print("courant:", sol.controls.courant_number())


def init(s, run="A_adiab"):
    sol = s.settings.solution
    sol.initialization.initialization_type = "hybrid"
    sol.initialization.hybrid_initialize()
    s.settings.file.write(file_type="case-data", file_name=(HERE / f"{run}_init").as_posix())
    print("initialised and saved", f"{run}_init.cas.h5 / .dat.h5")


def t300(s, run="A_T300", t_wall=300.0):
    """Isothermal walls at t_wall, restarted from the adiabatic solution in memory (no re-initialisation)."""
    bc = s.settings.setup.boundary_conditions.wall
    for z in WALLS:
        bc[z].thermal = {"thermal_condition": "Temperature", "temperature": {"value": t_wall}}
    sol = s.settings.solution
    surf = sol.report_definitions.surface
    surf["q-nose-avg"] = {"report_type": "surface-areaavg", "field": "heat-flux", "surface_names": ["wall_nose"]}
    surf["q-nose-min"] = {"report_type": "surface-facetmin", "field": "heat-flux", "surface_names": ["wall_nose"]}
    sol.monitor.report_files["cd-file"] = {"report_defs": ["cd-total", "cd-nose", "q-nose-avg", "q-nose-min"],
                                           "file_name": (HERE / f"{run}_cd.out").as_posix()}
    sol.monitor.report_plots["q-plot"] = {"report_defs": ["q-nose-avg", "q-nose-min"]}
    s.settings.file.auto_save.root_name = (HERE / "autosave" / run).as_posix()
    for z in WALLS:
        th = bc[z].thermal()
        print(z, th["thermal_condition"], th["temperature"])
    for name in ("q-nose-avg", "q-nose-min"):
        d = surf[name]()
        print(name, {k: d[k] for k in ("report_type", "field", "surface_names")})
    print("report file", sol.monitor.report_files["cd-file"]()["report_defs"],
          sol.monitor.report_files["cd-file"]()["file_name"])
    print("autosave root", s.settings.file.auto_save.root_name())
    print("courant", sol.controls.courant_number(), "schemes", sol.methods.discretization_scheme())


def all_(s):
    general(s)
    bcs(s)
    methods(s)


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
    """Study A at M 0.95: 200 iterations first order, then second order (1200 in total)."""
    it = s.settings.solution.run_calculation.iterate
    it(iter_count=first_iters)
    pb_order(s, "second")
    it(iter_count=second_iters)
    s.settings.file.write(file_type="case-data", file_name=(HERE / run).as_posix())
    print("schedule done, saved", run)


def schedule(s, run, second_order_iters=3000):
    """A_adiab's schedule: 500 first order at CFL 1, 500 at CFL 5, then second order at CFL 5."""
    it = s.settings.solution.run_calculation.iterate
    it(iter_count=500)
    methods(s, courant=5.0)
    it(iter_count=500)
    methods(s, order="second-order-upwind", courant=5.0)
    it(iter_count=second_order_iters)
    s.settings.file.write(file_type="case-data", file_name=(HERE / run).as_posix())
    print("schedule done, saved", run)


if __name__ == "__main__":
    s = connect()
    if sys.argv[1] == "schedule":
        schedule(s, sys.argv[2])
    else:
        {"general": general, "bcs": bcs, "methods": methods, "init": init, "all": all_, "t300": t300}[sys.argv[1]](s)

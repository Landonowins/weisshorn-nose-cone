"""Design study: wave drag -> flight performance, mass, payload fit, stability and heating.

:class:`Study` ties the configuration, the profiles and the CFD database together and provides
the parameter sweep (profiles x fineness x tip radii), the apogee sensitivities, the robustness
scenarios, the bluntness cost, the heating results and the weighted decision matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import aerodynamics as aero
from . import heating as ht
from .config import Config, parse_station
from .flight import DragModel, calibrate, simulate
from .geometry import (Profile, base_angle_deg, blunt_tangent_point, length_with_radius, make_profiles,
                       tip_half_angle_deg, volume_and_area)
from .stability import nose_mass as _nose_mass
from .stability import static_margin_shift


def mach_key(M: float) -> str:
    """Column suffix of a Mach number: 1.1 -> '11', 1.25 -> '125'."""
    return f"{M:g}".replace(".", "")


def matrix_weights(weights: dict) -> dict:
    """Decision-matrix weights normalised to add up to 1: {4, 2, 2, 1} -> {4/9, 2/9, 2/9, 1/9}."""
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


@dataclass
class Study:
    """Everything needed to evaluate noses for one configuration.

    Args:
        cfg: configuration.
        db: full CFD database (all studies).
        k: factor on the non-nose drag; calibrated if None and flight.calibrate is true.
    """

    cfg: Config
    db: pd.DataFrame
    k: Optional[float] = None

    def __post_init__(self) -> None:
        self.profiles: dict[str, Profile] = make_profiles(self.cfg.nose.custom_profiles)
        self.R = self.cfg.R
        if self.k is None:
            self.k = self.calibrated_k() if self.cfg.flight.calibrate else self.cfg.flight.drag_factor

    # ------------------------------------------------------------------ building blocks
    @cached_property
    def db_aero(self) -> pd.DataFrame:
        """Database rows used by the wave-drag model (configured studies and fineness range)."""
        return aero.select_from_config(self.db, self.cfg, rn=None)

    def L(self, fineness: float) -> float:
        """Nose length [m] for a fineness ratio."""
        return 2 * self.R * fineness

    def wave_drag(self, profile: str, fineness: float, rn: float = 0.0, onset: Optional[float] = None) -> aero.WaveDrag:
        """Wave-drag model of a profile (raises ValueError if the database has no data)."""
        df = self.db_aero if rn else aero.select_from_config(self.db, self.cfg, rn=0.0)
        return aero.wave_drag(self.cfg, df, profile, fineness, rn, onset)

    def vehicle(self, profile: str, fineness: float, k: Optional[float] = None, rn: float = 0.0,
                onset: Optional[float] = None) -> DragModel:
        """Drag model of the vehicle with a given nose."""
        L = self.L(fineness)
        _, S = volume_and_area(self.profiles[profile], L, self.R, rn * self.R)
        return DragModel(self.cfg, self.wave_drag(profile, fineness, rn, onset), S, L, self.k if k is None else k)

    def nose_mass(self, profile: str, fineness: float) -> float:
        """Shell + tip mass [kg]."""
        return _nose_mass(self.cfg, self.profiles[profile], self.L(fineness))

    @property
    def baseline(self) -> tuple[str, float]:
        """(profile, fineness) of the reference nose."""
        n = self.cfg.nose
        return n.baseline_profile, n.baseline_fineness

    @property
    def design(self) -> tuple[str, float, float]:
        """(profile, fineness, tip radius ratio) of the selected design."""
        n = self.cfg.nose
        return n.design_profile, n.design_fineness, n.design_tip_radius_ratio

    def mass_delta(self, profile: str, fineness: float) -> float:
        """Nose mass relative to the baseline nose [kg]."""
        return self.nose_mass(profile, fineness) - self.nose_mass(*self.baseline)

    def calibrated_k(self) -> float:
        """Factor k such that the baseline nose reaches flight.target_apogee."""
        return calibrate(self.cfg, lambda k: self.vehicle(*self.baseline, k=k))

    def nose_bay_length(self, profile: str, fineness: float) -> float:
        """Curved length + the cylinder the payload still needs behind the shoulder (>= the shoulder) [m]."""
        p = self.cfg.payload
        L = self.L(fineness)
        l_in = length_with_radius(self.profiles[profile], L, self.R, p.required_radius)
        return L + max(self.cfg.nose.shoulder_length, p.length + p.axial_clearance - l_in)

    # ---------------------------------------------------------------------- trajectories
    def trajectories(self) -> tuple[DragModel, dict, dict]:
        """Vehicle and the nominal and hot-day envelope trajectories used for heating and plots."""
        fl = self.cfg.flight
        if self.cfg.heating.trajectory_from == "baseline":
            veh = self.vehicle(*self.baseline)
        else:
            p, f, _ = self.design
            veh = self.vehicle(p, f)
        nom = simulate(self.cfg, veh, mass_model=fl.mass_model, dt_out=fl.detail_output_step)
        env = simulate(self.cfg, veh, mass_model=fl.envelope_mass_model, ground_temperature=fl.envelope_ground_temperature,
                       dt_out=fl.detail_output_step)
        return veh, nom, env

    # --------------------------------------------------------------------------- sweep
    def sweep(self, verbose: bool = True, log: Callable[[str], None] = print) -> pd.DataFrame:
        """Evaluate every profile x fineness x tip radius of the configuration.

        Rows without CFD data are skipped (with a message). Columns: geometry (L, volume, wetted
        area, mass, shoulder angle, CP), payload fit, wave drag at the tabulated Mach numbers,
        apogee, maximum Mach number, static-margin change, and changes relative to the baseline.
        """
        cfg, n = self.cfg, self.cfg.nose
        rns = list(n.tip_radius_ratios)
        with_rn = len(rns) > 1 or any(r > 0 for r in rns)
        rows = []
        for shape in n.profiles:
            pr = self.profiles[shape]
            for f in n.fineness:
                for rn in rns:
                    L = self.L(f)
                    V, S = volume_and_area(pr, L, self.R, rn * self.R)
                    try:
                        veh = self.vehicle(shape, f, rn=rn)
                    except ValueError as exc:
                        if verbose:
                            log(f"  skipped {shape} L/D={f:g} rn/R={rn:g}: {exc}")
                        continue
                    dm = self.mass_delta(shape, f)
                    tr = simulate(cfg, veh, m_nose_delta=dm, mass_model=cfg.flight.mass_model)
                    wd = veh.cdw
                    row = dict(shape=shape, fineness=f)
                    if with_rn:
                        row["tip_radius_ratio"] = rn
                    row.update(L_mm=1e3 * L, V_L=1e3 * V, S_m2=S, mass_g=1e3 * self.nose_mass(shape, f),
                               base_angle=base_angle_deg(pr, L, self.R),
                               xcp_over_L=1 - V / (np.pi * self.R**2 * L))
                    if cfg.payload.enabled:
                        row.update(l3U_mm=1e3 * length_with_radius(pr, L, self.R, cfg.payload.required_radius),
                                   bay_mm=1e3 * self.nose_bay_length(shape, f))
                    for M in cfg.aerodynamics.machs:
                        row[f"cd_w_{mach_key(M)}"] = float(wd(M))
                    row.update(apogee=tr["apogee"], mach_max=tr["M"].max(),
                               dSM_cal=static_margin_shift(cfg, pr, f, self.profiles[self.baseline[0]], self.baseline[1]))
                    rows.append(row)
        t = pd.DataFrame(rows)
        if t.empty:
            raise ValueError("no nose of the sweep has CFD data; check nose.profiles / nose.fineness or "
                             "set aerodynamics.compute_missing: true")
        b_row = self.baseline_row()
        t["d_apogee"] = t.apogee - b_row["apogee"]
        if cfg.payload.enabled:
            t["d_bay_mm"] = t.bay_mm - b_row["bay_mm"]
        return t

    def baseline_row(self) -> dict:
        """Apogee (and nose-bay length) of the baseline nose."""
        p, f = self.baseline
        tr = simulate(self.cfg, self.vehicle(p, f), m_nose_delta=0.0, mass_model=self.cfg.flight.mass_model)
        out = dict(apogee=tr["apogee"])
        if self.cfg.payload.enabled:
            out["bay_mm"] = 1e3 * self.nose_bay_length(p, f)
        return out

    def sensitivities(self) -> dict:
        """Apogee change per kg of mass and per unit C_D (whole flight), around the baseline."""
        fl = self.cfg.flight
        veh = self.vehicle(*self.baseline)
        z0 = simulate(self.cfg, veh, mass_model=fl.mass_model)["apogee"]
        z_m = simulate(self.cfg, veh, m_nose_delta=fl.mass_sensitivity_step, mass_model=fl.mass_model)["apogee"]
        veh2 = self.vehicle(*self.baseline)
        base_cdw = veh2.cdw
        veh2.cdw = lambda M: base_cdw(M) + fl.cd_sensitivity_step
        z_cd = simulate(self.cfg, veh2, mass_model=fl.mass_model)["apogee"]
        return dict(dz_dm=(z_m - z0) / fl.mass_sensitivity_step, dz_dcd=(z_cd - z0) / fl.cd_sensitivity_step,
                    dz_per_cd_step=z_cd - z0, cd_step=fl.cd_sensitivity_step, apogee_baseline=z0)

    # ---------------------------------------------------------------------- robustness
    def robustness(self, verbose: bool = True, log: Callable[[str], None] = print) -> pd.DataFrame:
        """Apogee of every profile at robustness.fineness under each scenario, relative to the reference."""
        rb, cfg = self.cfg.robustness, self.cfg
        F = rb.fineness
        rows = []
        for sc in rb.scenarios:
            z = {}
            for s in cfg.nose.profiles:
                try:
                    veh = self.vehicle(s, F, k=self.k * sc.get("drag_factor_scale", 1.0),
                                       onset=sc.get("transonic_onset"))
                except ValueError:
                    continue
                veh.cdw = aero.scaled(veh.cdw, sc.get("transonic_factor", 1.0), cfg.aerodynamics.transonic_match)
                z[s] = simulate(cfg, veh, m_nose_delta=self.mass_delta(s, F), mass_model=sc.get("mass_model"),
                                ground_temperature=sc.get("ground_temperature"))["apogee"]
            ref = z.get(rb.reference_profile)
            for s in z:
                rows.append(dict(scenario=sc["name"], shape=s, apogee=z[s],
                                 dz_vs_vk=z[s] - ref if ref is not None else np.nan))
            if verbose:
                log(f"  {sc['name']}: " + ", ".join(f"{s} {z[s] - ref:+.0f} m" for s in z if ref is not None))
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------- bluntness
    def bluntness(self) -> list[dict]:
        """Wave-drag and apogee cost of each reported tip radius (from the bluntness study)."""
        bl, cfg = self.cfg.bluntness, self.cfg
        d = aero.select(self.db, [bl.study], cfg.aerodynamics.min_fineness, cfg.aerodynamics.max_fineness, rn=None)
        d0 = aero.select_from_config(self.db, cfg, rn=0.0)
        base = d0[(d0["shape"] == bl.profile) & (d0.fineness == bl.fineness)].set_index("mach").cd
        z0 = simulate(cfg, self.vehicle(bl.profile, bl.fineness), mass_model=cfg.flight.mass_model)["apogee"]
        out = []
        for rnr in bl.report_ratios:
            b = d[(d["shape"] == bl.profile) & np.isclose(d.rn, rnr)].set_index("mach").cd
            Ms = np.array(sorted(set(b.index) & set(base.index)))
            if len(Ms) == 0 or bl.report_mach not in Ms:
                continue
            ratio = np.array([b.loc[M] / base.loc[M] for M in Ms])
            vb = self.vehicle(bl.profile, bl.fineness)
            cdw0 = vb.cdw
            vb.cdw = lambda M, c=cdw0, x=Ms, r=ratio: c(M) * np.interp(M, x, r)
            dz = simulate(cfg, vb, mass_model=cfg.flight.mass_model)["apogee"] - z0
            out.append(dict(rn_over_R=rnr, drag_cost_pct=100 * (b.loc[bl.report_mach] / base.loc[bl.report_mach] - 1),
                            d_apogee=dz))
        return out

    # ------------------------------------------------------------------------- heating
    def materials(self) -> dict[str, ht.Material]:
        """Materials of heating.materials as Material objects."""
        return {k: ht.Material.from_dict(k, v) for k, v in self.cfg.heating.materials.items()}

    def heating(self, nom: dict, env: dict) -> dict:
        """Peak temperatures [degC] of the tip insert, the printed tip and the shell (nominal, envelope)."""
        h, cfg = self.cfg.heating, self.cfg
        mats = self.materials()
        p, f, rnr = self.design
        pr = self.profiles[p]
        L, rn = self.L(f), rnr * self.R
        tip = [ht.tip_insert(tr, pr, L, self.R, rn, h.tip_length, cfg.structure.tip_mass, mats[h.tip_material])["T"].max()
               - 273.15 for tr in (nom, env)]
        printed = [ht.wall_response(tr, mats[h.printed_tip_material], h.printed_tip_thickness, "stagnation", rn=rn,
                                    n=h.wall_nodes)["T_out"].max() - 273.15 for tr in (nom, env)]
        stations = [parse_station(s, L) for s in h.shell_stations]
        shell = []
        for tr in (nom, env):
            shell.append(max(ht.wall_response(tr, mats[h.shell_material], h.shell_thickness, x, n=h.wall_nodes)["T_in"].max()
                             for x in stations) - 273.15)
        p0, T0 = ht.stagnation_state(nom["M"], nom["p"], nom["T"])
        p0e, T0e = ht.stagnation_state(env["M"], env["p"], env["T"])
        return dict(tip_insert=tip, printed_tip=printed, shell_inner=shell,
                    T0_peak=[float(np.max(T0)) - 273.15, float(np.max(T0e)) - 273.15],
                    mach_max=[float(nom["M"].max()), float(env["M"].max())])

    # -------------------------------------------------------------------- final design
    def design_definition(self, trade: pd.DataFrame) -> dict:
        """Geometry and performance of the selected design (blunted tip included)."""
        p, f, rnr = self.design
        pr = self.profiles[p]
        L, rn = self.L(f), rnr * self.R
        xt, xc = blunt_tangent_point(pr, L, self.R, rn)
        V, S = volume_and_area(pr, L, self.R, rn)
        out = dict(profile=p, fineness=f, tip_radius_ratio=rnr, L_m=L, tip_radius_m=rn,
                   L_physical_m=L - (xc - rn) if rn > 0 else L, truncation_m=(xc - rn) if rn > 0 else 0.0,
                   tangency_x_m=xt, volume_m3=V, wetted_area_m2=S, x_cp_m=L - V / (np.pi * self.R**2),
                   shoulder_angle_deg=base_angle_deg(pr, L, self.R))
        sel = trade[(trade["shape"] == p) & np.isclose(trade.fineness, f)]
        if len(sel):
            out["trade_row"] = sel.iloc[0]
        return out

    # ------------------------------------------------------------------- decision matrix
    def trade_matrix(self, trade: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Weighted decision matrix at trade_matrix.fineness (scores 1 = worst ... 5 = best)."""
        tm = self.cfg.trade_matrix
        order = list(self.cfg.nose.profiles)
        d = trade[np.isclose(trade.fineness, tm.fineness)].set_index("shape").reindex(order)
        if d.apogee.isna().all():
            return None

        def lin(v, hi_good=True):
            v = np.asarray(v, float)
            span = np.nanmax(v) - np.nanmin(v)
            s = (v - np.nanmin(v)) / span if span > 0 else np.zeros_like(v)
            return 1 + 4 * (s if hi_good else 1 - s)

        crit = {"performance": lin(d.d_apogee),
                "integration": lin(d.l3U_mm) if "l3U_mm" in d else np.full(len(order), np.nan),
                "manufacture": np.array([tm.manufacture.get(s, np.nan) for s in order], float),
                "stability": lin(d.dSM_cal)}
        w = matrix_weights(tm.weights)
        rows = {c: crit[c] for c in w}
        total = sum(w[c] * rows[c] for c in w)
        m = pd.DataFrame(rows, index=order).T
        m.loc["total"] = total
        return m

    # ----------------------------------------------------------------------- catalogue
    def catalogue(self, trade: pd.DataFrame, fineness_rows: tuple = (4.0, 4.5, 5.0)) -> pd.DataFrame:
        """Metrics of every profile at the design fineness (CFD values computed at that fineness if available)."""
        cfg = self.cfg
        f = cfg.nose.design_fineness
        L = self.L(f)
        d45 = aero.select_from_config(self.db, cfg, rn=0.0)
        d45 = d45[np.isclose(d45.fineness, f)]
        tt = trade.set_index(["shape", "fineness"])
        out = {}
        for s in cfg.nose.profiles:
            pr = self.profiles[s]
            V, S = volume_and_area(pr, L, self.R)
            col = dict(volume_L=V * 1e3, wetted_area_m2=S, tip_semi_angle_deg=tip_half_angle_deg(pr, L, self.R),
                       shoulder_angle_deg=abs(base_angle_deg(pr, L, self.R)))
            col["xcp_over_L"] = tt.loc[(s, f)].xcp_over_L if (s, f) in tt.index else np.nan
            for M in sorted(set(d45.mach)):
                v = d45[(d45["shape"] == s) & (d45.mach == M)].cd
                col[f"cd_w_direct_{mach_key(M)}"] = float(v.values[0]) if len(v) else np.nan
            for fr in fineness_rows:
                col[f"d_apogee_LD{fr:g}"] = tt.loc[(s, fr)].d_apogee if (s, fr) in tt.index else np.nan
            for key in ("mass_g", "l3U_mm", "bay_mm", "dSM_cal"):
                col[key] = tt.loc[(s, f)][key] if (s, f) in tt.index and key in tt.columns else np.nan
            out[s] = col
        return pd.DataFrame(out)

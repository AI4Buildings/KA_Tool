"""MCP server exposing the HVAC planning tool to LLMs."""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from src.thermodynamics import moist_air
from src.components.fan import Fan
from src.components.frost_protection import FrostProtectionPreheater, FrostProtectionBypass
from src.components.heat_recovery import (
    PlateHeatRecovery, RotaryHeatRecovery, RunaroundCoilRecovery,
)
from src.components.heating_coil import HeatingCoil
from src.components.cooling_coil import CoolingCoil
from src.components.humidifier import SprayHumidifier, SteamHumidifier
from src.system.ahu_system import AHUSystem
from src.thermodynamics.air_state import AirState
from src.system.simulation import HourlySimulation
from src.weather.open_meteo import OpenMeteoClient

mcp = FastMCP("hvac-planning-tool", instructions=(
    "HVAC planning tool for calculating energy demand of air handling units. "
    "Use calculate_single_timestep for a single operating point analysis, "
    "calculate_ahu_energy for annual simulations with weather data, "
    "compare_ahu_concepts to compare multiple configurations, and "
    "get_design_weather_data for typical design conditions at a location."
))

weather_client = OpenMeteoClient()


# ---------------------------------------------------------------------------
# Helper: build a Component instance from a config dict
# ---------------------------------------------------------------------------

def _build_component(name: str, config: dict):
    """Instantiate the correct Component class from a config dict.

    Args:
        name: Component key in the system_config (e.g. 'fan', 'hrv').
        config: Dict with at least a 'type' key plus constructor kwargs.

    Returns:
        A Component instance.
    """
    ctype = config.get("type", name)

    if ctype == "fan":
        return Fan(
            P_el_kW=config.get("P_el_kW", 0.0),
            heat_recovery_factor=config.get("heat_recovery_factor", 1.0),
        )

    if ctype == "frost_protection":
        variant = config.get("variant", "electric_preheater")
        limit_T = config.get("limit_temperature", -5.0)
        if variant == "bypass":
            return FrostProtectionBypass(limit_temperature=limit_T)
        return FrostProtectionPreheater(limit_temperature=limit_T)

    if ctype == "hrv":
        variant = config.get("variant", "plate")

        if variant == "plate":
            return PlateHeatRecovery(
                design_airflow_supply=config.get("design_airflow_supply", 5000.0),
                design_airflow_exhaust=config.get("design_airflow_exhaust", 5000.0),
                effectiveness_nominal=config.get("effectiveness_nominal", 0.6),
                design_T_outdoor=config.get("design_T_outdoor", -12.0),
                design_T_exhaust=config.get("design_T_exhaust", 22.0),
            )

        if variant == "rotary":
            return RotaryHeatRecovery(
                rotor_type=config.get("rotor_type", "ROT_SORP"),
                control_mode=config.get("optimization_mode", "Temp"),
                design_airflow_supply=config.get("design_airflow_supply", 5000.0),
                n_rot_max=config.get("n_rot_max", 20.0),
                f_ODA_min=config.get("f_ODA_min", 1.0),
                eta_hr_N=config.get("eta_hr_N", 0.0),
                eta_xr_N=config.get("eta_xr_N", 0.0),
            )

        if variant == "runaround":
            return RunaroundCoilRecovery(
                design_airflow_supply=config.get("design_airflow_supply", 5000.0),
                design_airflow_exhaust=config.get("design_airflow_exhaust", 5000.0),
                design_V_dot_medium=config.get("design_V_dot_medium", 1.0),
                effectiveness_supply=config.get("effectiveness_supply", 0.6),
                effectiveness_exhaust=config.get("effectiveness_exhaust", 0.6),
                design_T_outdoor=config.get("design_T_outdoor", -12.0),
                design_T_exhaust=config.get("design_T_exhaust", 22.0),
            )

        raise ValueError(f"Unknown HRV variant: {variant}")

    if ctype == "heating_coil":
        return HeatingCoil(
            coil_type=config.get("coil_type", "post"),
            design_capacity_kW=config.get("design_capacity_kW", 0.0),
            design_supply_T_water=config.get("design_supply_T_water", 60.0),
            design_return_T_water=config.get("design_return_T_water", 45.0),
            design_airflow=config.get("design_airflow", 5000.0),
            design_air_T_in=config.get("design_air_T_in", -12.0),
            design_air_T_out=config.get("design_air_T_out", 18.0),
        )

    if ctype == "cooling_coil":
        return CoolingCoil(
            dehumidification=config.get("dehumidification", True),
            design_capacity_kW=config.get("design_capacity_kW", 25.0),
            design_supply_T_water=config.get("design_supply_T_water", 6.0),
            design_return_T_water=config.get("design_return_T_water", 12.0),
            design_airflow=config.get("design_airflow", 5000.0),
            design_air_T_in=config.get("design_air_T_in", 28.0),
            design_air_phi_in=config.get("design_air_phi_in", 0.6),
            design_air_T_out=config.get("design_air_T_out", 13.0),
            design_air_phi_out=config.get("design_air_phi_out", 0.95),
        )

    if ctype == "humidifier":
        variant = config.get("variant", "steam")
        if variant == "spray":
            return SprayHumidifier(
                efficiency=config.get("efficiency", 0.9),
            )
        return SteamHumidifier(
            T_steam=config.get("T_steam", 100.0),
        )

    if ctype == "adiabatic_cooling":
        # Adiabatic cooling uses SprayHumidifier on exhaust side
        return SprayHumidifier(
            efficiency=config.get("efficiency", 0.9),
        )

    raise ValueError(f"Unknown component type: {ctype}")


def _build_ahu_system(system_config: dict,
                      airflow_supply: float,
                      airflow_exhaust: float,
                      supply_temp_setpoint: float,
                      supply_humidity_setpoint_x: float) -> AHUSystem:
    """Build an AHUSystem from a system_config dict."""
    components_cfg = system_config.get("components", {})

    supply_components = []
    for comp_name in system_config.get("supply_air", []):
        cfg = components_cfg.get(comp_name, {"type": comp_name})
        supply_components.append(_build_component(comp_name, cfg))

    exhaust_components = []
    for comp_name in system_config.get("exhaust_air", []):
        cfg = components_cfg.get(comp_name, {"type": comp_name})
        exhaust_components.append(_build_component(comp_name, cfg))

    return AHUSystem(
        supply_air_components=supply_components,
        exhaust_air_components=exhaust_components,
        design_airflow_supply=airflow_supply,
        design_airflow_exhaust=airflow_exhaust,
        setpoint_T=supply_temp_setpoint,
        setpoint_x=supply_humidity_setpoint_x,
    )


def _resolve_supply_setpoints(
    room_temp_setpoint: float,
    room_humidity_setpoint: float,
    supply_temp_setpoint: Optional[float],
    supply_humidity_setpoint: Optional[float],
    p: float = 101325.0,
) -> tuple[float, float]:
    """Resolve supply air setpoints from room setpoints if not given.

    Returns:
        (T_supply_setpoint [deg C], x_supply_setpoint [kg/kg])
    """
    T_supply = supply_temp_setpoint if supply_temp_setpoint is not None else room_temp_setpoint
    if supply_humidity_setpoint is not None:
        x_supply = moist_air.absolute_humidity(supply_humidity_setpoint / 100.0, T_supply, p)
    else:
        x_supply = moist_air.absolute_humidity(room_humidity_setpoint / 100.0, room_temp_setpoint, p)
    return T_supply, x_supply


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate_single_timestep(
    outdoor_T: float,
    outdoor_phi: float,
    exhaust_T: float,
    exhaust_phi: float,
    airflow_supply: float,
    airflow_exhaust: float,
    supply_temp_setpoint: float,
    supply_humidity_setpoint: float,
    system_config: dict,
    outdoor_p: float = 101325.0,
) -> dict[str, Any]:
    """Calculate a single timestep for an AHU with fixed boundary conditions.

    Use this tool to analyse a specific operating point without weather data.
    Ideal for design-point checks and component-level debugging.

    Args:
        outdoor_T: Outdoor air temperature [deg C].
        outdoor_phi: Outdoor air relative humidity [% RH, 0-100].
        exhaust_T: Exhaust (room) air temperature [deg C].
        exhaust_phi: Exhaust (room) air relative humidity [% RH, 0-100].
        airflow_supply: Supply airflow [m3/h].
        airflow_exhaust: Exhaust airflow [m3/h].
        supply_temp_setpoint: Supply air temperature setpoint [deg C].
        supply_humidity_setpoint: Supply air humidity setpoint [% RH, 0-100].
        system_config: AHU configuration dict with keys "supply_air",
            "exhaust_air", "components".
        outdoor_p: Atmospheric pressure [Pa]. Default 101325.

    Returns:
        Dict with supply air outlet state, energy totals, and per-component
        breakdown.
    """
    # Convert RH to absolute humidity
    x_outdoor = moist_air.absolute_humidity(outdoor_phi / 100.0, outdoor_T, outdoor_p)
    x_exhaust = moist_air.absolute_humidity(exhaust_phi / 100.0, exhaust_T, outdoor_p)
    x_supply_setpoint = moist_air.absolute_humidity(
        supply_humidity_setpoint / 100.0, supply_temp_setpoint, outdoor_p)

    outdoor_air = AirState(T=outdoor_T, x=x_outdoor, p=outdoor_p)
    exhaust_air = AirState(T=exhaust_T, x=x_exhaust, p=outdoor_p)

    ahu = _build_ahu_system(
        system_config, airflow_supply, airflow_exhaust,
        supply_temp_setpoint, x_supply_setpoint,
    )

    result = ahu.calculate_timestep(
        outdoor_air=outdoor_air,
        exhaust_air=exhaust_air,
        V_dot_supply=airflow_supply,
        V_dot_exhaust=airflow_exhaust,
        timestep_h=1.0,
    )

    # Build per-component breakdown
    component_details = {}
    for name, cr in result.component_results.items():
        detail = {
            "T_out": round(cr.air_out.T, 2),
            "x_out_g_kg": round(cr.air_out.x * 1000, 3),
            "phi_out_pct": round(cr.air_out.phi * 100, 1),
        }
        if cr.Q_heat != 0:
            detail["Q_heat_kWh"] = round(cr.Q_heat, 3)
        if cr.Q_cool != 0:
            detail["Q_cool_kWh"] = round(cr.Q_cool, 3)
        if cr.W_el != 0:
            detail["W_el_kWh"] = round(cr.W_el, 3)
        if cr.m_water != 0:
            detail["m_water_kg"] = round(cr.m_water, 3)
        if cr.info:
            detail["info"] = cr.info
        component_details[name] = detail

    return {
        "supply_air_out": {
            "T": round(result.air_supply_out.T, 2),
            "x_g_kg": round(result.air_supply_out.x * 1000, 3),
            "phi_pct": round(result.air_supply_out.phi * 100, 1),
            "h_kJ_kg": round(result.air_supply_out.h, 2),
        },
        "outdoor_air_in": {
            "T": outdoor_T,
            "x_g_kg": round(x_outdoor * 1000, 3),
            "phi_pct": outdoor_phi,
        },
        "totals": {
            "Q_heat_kWh": round(result.Q_heat, 3),
            "Q_cool_kWh": round(result.Q_cool, 3),
            "W_el_kWh": round(result.W_el, 3),
            "m_water_kg": round(result.m_water, 3),
        },
        "component_results": component_details,
    }


@mcp.tool()
def calculate_ahu_energy(
    location: str,
    start_date: str,
    end_date: str,
    airflow_supply: float,
    airflow_exhaust: float,
    room_temp_setpoint: float,
    room_humidity_setpoint: float,
    system_config: dict,
    supply_temp_setpoint: float | None = None,
    supply_humidity_setpoint: float | None = None,
) -> dict[str, Any]:
    """Calculate energy demand of an AHU for a defined period and location.

    Args:
        location: City name (e.g. "Wien") or "lat,lon" (e.g. "48.2,16.3").
        start_date: Start date "YYYY-MM-DD".
        end_date: End date "YYYY-MM-DD".
        airflow_supply: Supply airflow [m3/h].
        airflow_exhaust: Exhaust airflow [m3/h].
        room_temp_setpoint: Room temperature setpoint [deg C].
        room_humidity_setpoint: Room relative humidity setpoint [% RH].
        system_config: AHU configuration dict with keys "supply_air", "exhaust_air", "components".
        supply_temp_setpoint: Supply air temperature setpoint [deg C]. Derived from room setpoint if omitted.
        supply_humidity_setpoint: Supply air humidity setpoint [% RH]. Derived from room setpoint if omitted.

    Returns:
        Energy summary with totals, monthly breakdown, and peak loads.
    """
    # Resolve setpoints
    T_supply, x_supply = _resolve_supply_setpoints(
        room_temp_setpoint, room_humidity_setpoint,
        supply_temp_setpoint, supply_humidity_setpoint,
    )

    # Geocode and fetch weather data
    lat, lon, resolved_name = weather_client.geocode(location)
    weather_df = weather_client.get_hourly_weather(lat, lon, start_date, end_date)

    # Build AHU system
    ahu = _build_ahu_system(system_config, airflow_supply, airflow_exhaust, T_supply, x_supply)

    # Run hourly simulation
    sim = HourlySimulation(
        ahu=ahu,
        exhaust_air_T=room_temp_setpoint,
        exhaust_air_phi=room_humidity_setpoint / 100.0,
    )
    sim_result = sim.run(
        weather_data=weather_df,
        V_dot_supply=airflow_supply,
        V_dot_exhaust=airflow_exhaust,
    )

    return {
        "location": resolved_name,
        "period": f"{start_date} to {end_date}",
        "Q_heat_total_kWh": round(sim_result.total_Q_heat_kWh, 1),
        "Q_cool_total_kWh": round(sim_result.total_Q_cool_kWh, 1),
        "W_el_total_kWh": round(sim_result.total_W_el_kWh, 1),
        "m_water_total_kg": round(sim_result.total_m_water_kg, 1),
        "operating_hours": len(sim_result.hourly_results),
        "peak_loads": sim_result.peak_loads,
        "monthly_breakdown": sim_result.monthly_breakdown,
    }


@mcp.tool()
def compare_ahu_concepts(
    concepts: list[dict],
    concept_names: list[str],
    location: str,
    start_date: str,
    end_date: str,
    room_conditions: dict,
) -> dict[str, Any]:
    """Compare multiple AHU concepts under the same boundary conditions.

    Args:
        concepts: List of system configuration dicts. Each dict must contain
            "system_config", and optionally "airflow_supply" / "airflow_exhaust".
        concept_names: Names for each concept (same length as concepts).
        location: City name or "lat,lon".
        start_date: Start date "YYYY-MM-DD".
        end_date: End date "YYYY-MM-DD".
        room_conditions: Dict with "room_temp_setpoint" [deg C],
            "room_humidity_setpoint" [% RH], and optionally
            "airflow_supply" [m3/h], "airflow_exhaust" [m3/h],
            "supply_temp_setpoint" [deg C], "supply_humidity_setpoint" [% RH].

    Returns:
        Comparison table and recommendation text.
    """
    # Shared parameters from room_conditions
    room_T = room_conditions["room_temp_setpoint"]
    room_phi = room_conditions["room_humidity_setpoint"]
    default_supply = room_conditions.get("airflow_supply", 5000.0)
    default_exhaust = room_conditions.get("airflow_exhaust", default_supply)
    supply_T_sp = room_conditions.get("supply_temp_setpoint")
    supply_phi_sp = room_conditions.get("supply_humidity_setpoint")

    # Geocode and fetch weather once for all concepts
    lat, lon, resolved_name = weather_client.geocode(location)
    weather_df = weather_client.get_hourly_weather(lat, lon, start_date, end_date)

    T_supply, x_supply = _resolve_supply_setpoints(
        room_T, room_phi, supply_T_sp, supply_phi_sp,
    )

    comparison_table = []

    for concept_cfg, cname in zip(concepts, concept_names):
        sys_cfg = concept_cfg.get("system_config", concept_cfg)
        af_supply = concept_cfg.get("airflow_supply", default_supply)
        af_exhaust = concept_cfg.get("airflow_exhaust", default_exhaust)

        ahu = _build_ahu_system(sys_cfg, af_supply, af_exhaust, T_supply, x_supply)

        sim = HourlySimulation(
            ahu=ahu,
            exhaust_air_T=room_T,
            exhaust_air_phi=room_phi / 100.0,
        )
        sim_result = sim.run(
            weather_data=weather_df,
            V_dot_supply=af_supply,
            V_dot_exhaust=af_exhaust,
        )

        comparison_table.append({
            "concept_name": cname,
            "Q_heat_total_kWh": round(sim_result.total_Q_heat_kWh, 1),
            "Q_cool_total_kWh": round(sim_result.total_Q_cool_kWh, 1),
            "W_el_total_kWh": round(sim_result.total_W_el_kWh, 1),
            "m_water_total_kg": round(sim_result.total_m_water_kg, 1),
            "peak_loads": sim_result.peak_loads,
        })

    # Build recommendation
    if comparison_table:
        best = min(comparison_table, key=lambda r: (
            r.get("Q_heat_total_kWh", 0)
            + r.get("Q_cool_total_kWh", 0)
            + r.get("W_fan_total_kWh", 0)
            + r.get("W_humidifier_kWh", 0)
        ))
        recommendation = (
            f"Concept '{best['concept_name']}' has the lowest total energy demand."
        )
    else:
        recommendation = "No concepts provided."

    return {
        "comparison_table": comparison_table,
        "recommendation": recommendation,
    }


@mcp.tool()
def get_design_weather_data(
    location: str,
    year: int | None = None,
) -> dict[str, Any]:
    """Return typical design weather data for a location.

    Fetches a full year of hourly data and derives summer/winter design
    conditions and annual statistics.

    Args:
        location: City name or "lat,lon".
        year: Year to analyse. Defaults to previous year for a complete dataset.

    Returns:
        Dict with summer_design, winter_design, and annual_stats.
    """
    import datetime

    if year is None:
        year = datetime.date.today().year - 1

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    lat, lon, resolved_name = weather_client.geocode(location)
    weather_df = weather_client.get_hourly_weather(lat, lon, start_date, end_date)

    T = weather_df["T"]
    phi = weather_df["phi"]

    # Summer design: 99th percentile
    summer_T = float(T.quantile(0.99))
    summer_phi_values = phi[T >= T.quantile(0.95)]
    summer_phi = float(summer_phi_values.mean()) if len(summer_phi_values) > 0 else float(phi.mean())

    # Winter design: 1st percentile
    winter_T = float(T.quantile(0.01))
    winter_phi_values = phi[T <= T.quantile(0.05)]
    winter_phi = float(winter_phi_values.mean()) if len(winter_phi_values) > 0 else float(phi.mean())

    # Annual statistics
    hours_below_zero = int((T < 0).sum())
    hours_above_25 = int((T > 25).sum())

    return {
        "location": location,
        "year": year,
        "summer_design": {
            "T": round(summer_T, 1),
            "phi": round(summer_phi, 1),
        },
        "winter_design": {
            "T": round(winter_T, 1),
            "phi": round(winter_phi, 1),
        },
        "annual_stats": {
            "T_mean": round(float(T.mean()), 1),
            "T_min": round(float(T.min()), 1),
            "T_max": round(float(T.max()), 1),
            "phi_mean": round(float(phi.mean()), 1),
            "hours_below_0C": hours_below_zero,
            "hours_above_25C": hours_above_25,
            "total_hours": len(T),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

"""Cooling coil component with optional dehumidification."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air


class CoolingCoil(Component):
    """Cooling coil (Kühlregister) with optional dehumidification.

    Operating modes:
    - If x > x_soll and dehumidification active: cool + dehumidify to x_soll
    - If x <= x_soll: cool to T_soll
    - Dehumidification limit: water supply temperature

    Attributes:
        dehumidification: Whether dehumidification is enabled
        design_capacity_kW: Design cooling capacity [kW]
        design_supply_T_water: Design chilled water supply temperature [°C]
        design_return_T_water: Design chilled water return temperature [°C]
        design_airflow: Design air volume flow [m³/h]
    """

    def __init__(self, dehumidification: bool = True,
                 design_capacity_kW: float = 25.0,
                 design_supply_T_water: float = 6.0,
                 design_return_T_water: float = 12.0,
                 design_airflow: float = 5000.0,
                 design_air_T_in: float = 28.0,
                 design_air_phi_in: float = 0.6,
                 design_air_T_out: float = 13.0,
                 design_air_phi_out: float = 0.95):
        self.dehumidification = dehumidification
        self.design_capacity_kW = design_capacity_kW
        self.design_supply_T_water = design_supply_T_water
        self.design_return_T_water = design_return_T_water
        self.design_airflow = design_airflow
        self.design_air_T_in = design_air_T_in
        self.design_air_phi_in = design_air_phi_in
        self.design_air_T_out = design_air_T_out
        self.design_air_phi_out = design_air_phi_out

    @property
    def name(self) -> str:
        return "cooling_coil"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  dT_fan_correction: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate cooling coil output.

        Args:
            air_in: Inlet air state
            setpoint_T: Target temperature [°C]
            setpoint_x: Target absolute humidity [kg/kg]
            timestep_h: Time step [h]
            m_dot_dry: Dry air mass flow rate [kg/s]
            dT_fan_correction: Temperature offset from downstream fan [K]

        Returns:
            ComponentResult with cooled (and possibly dehumidified) air
        """
        if m_dot_dry <= 0:
            return ComponentResult(air_out=air_in.copy())

        T_target = setpoint_T - dT_fan_correction

        # Check if cooling is needed
        if air_in.T <= T_target and air_in.x <= setpoint_x:
            return ComponentResult(air_out=air_in.copy())

        h_in = moist_air.enthalpy(air_in.T, air_in.x)

        # Case 1: Dehumidification needed
        if self.dehumidification and air_in.x > setpoint_x:
            # Cool to dew point of target humidity, then along saturation line
            T_dew_target = moist_air.dew_point(setpoint_x, air_in.p)

            # Minimum achievable temperature is limited by water supply
            T_min = max(T_dew_target, self.design_supply_T_water + 1.0)

            # After dehumidification, air leaves at near saturation
            x_out = min(air_in.x, moist_air.saturation_humidity(T_min, air_in.p))
            x_out = max(x_out, setpoint_x)  # Don't dehumidify more than needed

            # Find the actual dew point for x_out
            T_sat = moist_air.dew_point(x_out, air_in.p)
            T_out = max(T_sat, T_min)

            # If target T is higher than the dehumidification temperature,
            # the post-heating coil will handle the reheat
            if T_target > T_out:
                pass  # NHR will reheat

        # Case 2: Sensible cooling only
        elif air_in.T > T_target:
            T_out = T_target
            x_out = air_in.x
            # Check for condensation
            x_sat_out = moist_air.saturation_humidity(T_out, air_in.p)
            if x_out > x_sat_out:
                x_out = x_sat_out  # Condensation occurs
        else:
            return ComponentResult(air_out=air_in.copy())

        air_out = AirState(T=T_out, x=x_out, p=air_in.p)
        h_out = moist_air.enthalpy(T_out, x_out)

        Q_dot = m_dot_dry * (h_in - h_out)  # kW (positive = cooling)
        Q_cool = Q_dot * timestep_h  # kWh

        # Condensate
        m_condensate = m_dot_dry * max(0, air_in.x - x_out) * timestep_h * 3600  # kg

        return ComponentResult(
            air_out=air_out,
            Q_cool=Q_cool,
            info={
                "Q_dot_kW": Q_dot,
                "m_condensate_kg": m_condensate,
                "dehumidification_active": x_out < air_in.x,
            }
        )

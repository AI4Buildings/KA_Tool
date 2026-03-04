"""Heating coil component (pre-heating and post-heating register)."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air


class HeatingCoil(Component):
    """Heating coil (Heizregister) - VHR or NHR.

    Heats air to the required temperature setpoint. The target temperature
    depends on downstream components (humidifier, cooling coil, fan).

    For pre-heating coil (VHR):
        - If humidification needed and steam humidifier: heat so that after
          humidification (T_soll, x_soll) is reached
        - If humidification needed and spray humidifier: heat so that after
          humidification x_soll is reached
        - Otherwise: heat to T_soll

    For post-heating coil (NHR):
        - Heat to T_soll if T_in < T_soll

    Attributes:
        coil_type: 'pre' or 'post'
        design_capacity_kW: Design heating capacity [kW]
        design_supply_T_water: Design water supply temperature [°C]
        design_return_T_water: Design water return temperature [°C]
        design_airflow: Design air volume flow [m³/h]
        design_air_T_in: Design air inlet temperature [°C]
        design_air_T_out: Design air outlet temperature [°C]
    """

    def __init__(self, coil_type: str = 'post',
                 design_capacity_kW: float = 0.0,
                 design_supply_T_water: float = 60.0,
                 design_return_T_water: float = 45.0,
                 design_airflow: float = 5000.0,
                 design_air_T_in: float = -12.0,
                 design_air_T_out: float = 18.0):
        self.coil_type = coil_type
        self.design_capacity_kW = design_capacity_kW
        self.design_supply_T_water = design_supply_T_water
        self.design_return_T_water = design_return_T_water
        self.design_airflow = design_airflow
        self.design_air_T_in = design_air_T_in
        self.design_air_T_out = design_air_T_out

    @property
    def name(self) -> str:
        return f"heating_coil_{self.coil_type}"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  dT_fan_correction: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate heating coil output.

        Args:
            air_in: Inlet air state
            setpoint_T: Target temperature [°C]
            setpoint_x: Target absolute humidity [kg/kg]
            timestep_h: Time step duration [h]
            m_dot_dry: Dry air mass flow rate [kg/s]
            dT_fan_correction: Temperature offset from downstream fan [K]

        Returns:
            ComponentResult with heated air state
        """
        if m_dot_dry <= 0:
            return ComponentResult(air_out=air_in.copy())

        # Target temperature (corrected for downstream fan heat)
        T_target = setpoint_T - dT_fan_correction

        if air_in.T >= T_target:
            # No heating needed
            return ComponentResult(air_out=air_in.copy())

        # Heat to target
        T_out = T_target
        air_out = AirState(T=T_out, x=air_in.x, p=air_in.p)

        h_in = moist_air.enthalpy(air_in.T, air_in.x)
        h_out = moist_air.enthalpy(T_out, air_in.x)
        Q_dot = m_dot_dry * (h_out - h_in)  # kW (positive = heating)

        # Limit to design capacity
        if self.design_capacity_kW > 0 and Q_dot > self.design_capacity_kW:
            Q_dot = self.design_capacity_kW
            h_out_limited = h_in + Q_dot / m_dot_dry
            T_out = moist_air.temperature_from_h_x(h_out_limited, air_in.x)
            air_out = AirState(T=T_out, x=air_in.x, p=air_in.p)

        Q_heat = Q_dot * timestep_h  # kWh

        return ComponentResult(
            air_out=air_out,
            Q_heat=Q_heat,
            info={"Q_dot_kW": Q_dot, "T_target": T_target}
        )

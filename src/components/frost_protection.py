"""Frost protection component - electric preheater or bypass."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air


class FrostProtectionPreheater(Component):
    """Electric frost protection preheater.

    Heats air to the limit temperature when inlet is below threshold.

    Attributes:
        limit_temperature: Activation threshold [°C]
            Typical: 0°C for plate HX, -5°C for rotary HX, +5°C for KVS
    """

    def __init__(self, limit_temperature: float = -5.0):
        self.limit_temperature = limit_temperature

    @property
    def name(self) -> str:
        return "frost_protection_preheater"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  **kwargs) -> ComponentResult:
        if air_in.T >= self.limit_temperature or m_dot_dry <= 0:
            return ComponentResult(air_out=air_in.copy())

        T_out = self.limit_temperature
        air_out = AirState(T=T_out, x=air_in.x, p=air_in.p)

        h_in = moist_air.enthalpy(air_in.T, air_in.x)
        h_out = moist_air.enthalpy(T_out, air_in.x)
        Q_dot = m_dot_dry * (h_out - h_in)  # kW
        W_el = Q_dot * timestep_h  # kWh (electric preheater)

        return ComponentResult(
            air_out=air_out,
            W_el=W_el,
            info={"dT_frost": T_out - air_in.T}
        )


class FrostProtectionBypass(Component):
    """Frost protection bypass.

    When active, bypasses the heat recovery unit.
    The preheating coil then handles the full heating load.

    Attributes:
        limit_temperature: Activation threshold [°C]
    """

    def __init__(self, limit_temperature: float = 0.0):
        self.limit_temperature = limit_temperature
        self.bypass_active = False

    @property
    def name(self) -> str:
        return "frost_protection_bypass"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  **kwargs) -> ComponentResult:
        self.bypass_active = air_in.T < self.limit_temperature
        return ComponentResult(
            air_out=air_in.copy(),
            info={"bypass_active": self.bypass_active}
        )

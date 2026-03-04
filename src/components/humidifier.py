"""Humidifier components - spray and steam."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air


class SprayHumidifier(Component):
    """Spray humidifier (adiabatic humidification).

    Humidifies air along approximately constant enthalpy lines.
    Limited by saturation (cannot exceed 100% RH).

    Attributes:
        efficiency: Humidification efficiency [0..1], default 0.9 (ÖNORM)
        T_water: Water temperature [°C], default 15
    """

    def __init__(self, efficiency: float = 0.9, T_water: float = 15.0):
        self.efficiency = efficiency
        self.T_water = T_water

    @property
    def name(self) -> str:
        return "spray_humidifier"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  **kwargs) -> ComponentResult:
        if m_dot_dry <= 0 or air_in.x >= setpoint_x:
            return ComponentResult(air_out=air_in.copy())

        h_water = moist_air.water_enthalpy(self.T_water)
        h_in = moist_air.enthalpy(air_in.T, air_in.x)

        # Find the saturation limit along the process line.
        # Process: x_out = x_in + dx, h_out = h_in + dx * h_water
        #          T_out = f(h_out, x_out)
        # Constraint: x_out <= x_sat(T_out)
        # We find dx_sat where x_in + dx_sat = x_sat(T(h_in + dx_sat * h_water, x_in + dx_sat))
        dx_sat = self._find_saturation_dx(air_in.T, air_in.x, h_in, h_water, air_in.p)

        # Apply humidification efficiency
        dx_max = dx_sat * self.efficiency

        # Required humidification
        dx_required = setpoint_x - air_in.x
        dx = min(dx_required, dx_max)

        if dx <= 0:
            return ComponentResult(air_out=air_in.copy())

        x_out = air_in.x + dx
        h_out = h_in + dx * h_water
        T_out = moist_air.temperature_from_h_x(h_out, x_out)

        # Safety check: clamp to saturation at outlet temperature
        x_sat_out = moist_air.saturation_humidity(T_out, air_in.p)
        if x_out > x_sat_out:
            x_out = x_sat_out
            T_out = moist_air.dew_point(x_out, air_in.p)

        air_out = AirState(T=T_out, x=x_out, p=air_in.p)

        # Water consumption [kg]
        dx_actual = x_out - air_in.x
        m_water = m_dot_dry * dx_actual * timestep_h * 3600

        return ComponentResult(
            air_out=air_out,
            m_water=m_water,
            info={"dx": dx_actual, "T_drop": air_in.T - T_out}
        )

    @staticmethod
    def _find_saturation_dx(T_in: float, x_in: float, h_in: float,
                            h_water: float, p: float) -> float:
        """Find dx where the process line hits the saturation curve.

        Uses bisection: at dx=0 we are unsaturated (x_in < x_sat(T_in)),
        and at some large dx the air would be supersaturated.
        """
        from scipy.optimize import brentq

        def residual(dx: float) -> float:
            x_test = x_in + dx
            h_test = h_in + dx * h_water
            T_test = moist_air.temperature_from_h_x(h_test, x_test)
            x_sat = moist_air.saturation_humidity(T_test, p)
            return x_sat - x_test  # positive = still unsaturated

        # Upper bracket: use a large dx that is certainly oversaturated
        dx_upper = 0.030  # 30 g/kg should always overshoot
        # Verify bracket
        if residual(0.0) <= 0:
            return 0.0  # Already saturated at inlet
        if residual(dx_upper) >= 0:
            return dx_upper  # Can't reach saturation (shouldn't happen)

        dx_sat = float(brentq(residual, 0.0, dx_upper, xtol=1e-7))
        return dx_sat


class SteamHumidifier(Component):
    """Steam humidifier (isothermal humidification).

    Adds steam to the air stream, increasing humidity with minimal
    temperature change (near-isothermal process).

    Attributes:
        T_steam: Steam temperature [°C], default 100
    """

    def __init__(self, T_steam: float = 100.0):
        self.T_steam = T_steam

    @property
    def name(self) -> str:
        return "steam_humidifier"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  **kwargs) -> ComponentResult:
        if m_dot_dry <= 0 or air_in.x >= setpoint_x:
            return ComponentResult(air_out=air_in.copy())

        dx = setpoint_x - air_in.x
        x_out = air_in.x + dx

        # Enthalpy of steam
        h_steam = moist_air.steam_enthalpy(self.T_steam)

        # New enthalpy
        h_in = moist_air.enthalpy(air_in.T, air_in.x)
        h_out = h_in + dx * h_steam
        T_out = moist_air.temperature_from_h_x(h_out, x_out)

        air_out = AirState(T=T_out, x=x_out, p=air_in.p)

        # Water consumption [kg]
        m_water = m_dot_dry * dx * timestep_h * 3600

        # Electrical energy for steam generation (approximate: 0.75 kWh/kg steam)
        W_el = m_water * 0.75  # kWh

        return ComponentResult(
            air_out=air_out,
            W_el=W_el,
            m_water=m_water,
            info={"dx": dx, "h_steam": h_steam}
        )

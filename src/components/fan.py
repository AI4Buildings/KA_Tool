"""Fan component - models temperature rise from fan motor heat."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air


class Fan(Component):
    """Fan component that adds heat to the air stream.

    The fan motor converts electrical energy to heat, a fraction of which
    is transferred to the air stream (depending on motor location).

    Attributes:
        heat_recovery_factor: Fraction of electrical power warming the air [0..1]
            0.0: no temperature rise (motor outside duct)
            0.6: motor outside duct (ÖNORM EN 16798-5-1:2017, Table B.12)
            1.0: motor inside duct
        P_el_kW: Electrical power consumption [kW]
    """

    def __init__(self, heat_recovery_factor: float = 1.0, P_el_kW: float = 0.0):
        self.heat_recovery_factor = heat_recovery_factor
        self.P_el_kW = P_el_kW

    @property
    def name(self) -> str:
        return "fan"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate temperature rise from fan.

        Args:
            air_in: Inlet air state
            setpoint_T: Not used for fan
            setpoint_x: Not used for fan
            timestep_h: Time step duration [h]
            m_dot_dry: Dry air mass flow rate [kg/s]

        Returns:
            ComponentResult with heated air and electrical energy
        """
        if m_dot_dry <= 0 or self.P_el_kW <= 0:
            return ComponentResult(air_out=air_in.copy())

        # Heat input to air [kW]
        Q_fan = self.P_el_kW * self.heat_recovery_factor

        # Temperature rise: Q = m_dot * c_p * dT
        c_p = 1.006  # kJ/(kg·K)
        dT = Q_fan / (m_dot_dry * c_p)

        air_out = AirState(T=air_in.T + dT, x=air_in.x, p=air_in.p)

        W_el = self.P_el_kW * timestep_h  # kWh

        return ComponentResult(
            air_out=air_out,
            W_el=W_el,
            info={"dT_fan": dT, "P_el_kW": self.P_el_kW}
        )

    def temperature_rise(self, m_dot_dry: float) -> float:
        """Calculate temperature rise [K] for given mass flow.

        Args:
            m_dot_dry: Dry air mass flow rate [kg/s]

        Returns:
            Temperature rise [K]
        """
        if m_dot_dry <= 0 or self.P_el_kW <= 0:
            return 0.0
        Q_fan = self.P_el_kW * self.heat_recovery_factor
        c_p = 1.006
        return Q_fan / (m_dot_dry * c_p)

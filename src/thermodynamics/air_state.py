"""AirState dataclass for representing the thermodynamic state of moist air."""

from dataclasses import dataclass, field
from src.thermodynamics import moist_air


@dataclass
class AirState:
    """Thermodynamic state of moist air.

    Primary state variables: T, x, p
    Derived properties (phi, h, T_dew, x_sat) are computed on access.

    Attributes:
        T: Temperature [°C]
        x: Absolute humidity [kg_H2O/kg_dry_air]
        p: Atmospheric pressure [Pa]
    """
    T: float
    x: float
    p: float = 101325.0

    @property
    def phi(self) -> float:
        """Relative humidity [0..1]."""
        return moist_air.relative_humidity(self.x, self.T, self.p)

    @property
    def h(self) -> float:
        """Specific enthalpy [kJ/kg_dry_air]."""
        return moist_air.enthalpy(self.T, self.x)

    @property
    def T_dew(self) -> float:
        """Dew point temperature [°C]."""
        return moist_air.dew_point(self.x, self.p)

    @property
    def x_sat(self) -> float:
        """Saturation absolute humidity at current temperature [kg/kg]."""
        return moist_air.saturation_humidity(self.T, self.p)

    @property
    def rho(self) -> float:
        """Density of moist air [kg/m³]."""
        return moist_air.density_moist_air(self.T, self.x, self.p)

    def copy(self) -> 'AirState':
        """Return a copy of this state."""
        return AirState(T=self.T, x=self.x, p=self.p)

    @classmethod
    def from_T_phi(cls, T: float, phi: float, p: float = 101325.0) -> 'AirState':
        """Create AirState from temperature and relative humidity.

        Args:
            T: Temperature [°C]
            phi: Relative humidity [0..1]
            p: Atmospheric pressure [Pa]
        """
        x = moist_air.absolute_humidity(phi, T, p)
        return cls(T=T, x=x, p=p)

    @classmethod
    def from_h_x(cls, h: float, x: float, p: float = 101325.0) -> 'AirState':
        """Create AirState from enthalpy and absolute humidity.

        Args:
            h: Specific enthalpy [kJ/kg_dry_air]
            x: Absolute humidity [kg_H2O/kg_dry_air]
            p: Atmospheric pressure [Pa]
        """
        T = moist_air.temperature_from_h_x(h, x)
        return cls(T=T, x=x, p=p)

    def __repr__(self) -> str:
        return (f"AirState(T={self.T:.1f}°C, x={self.x*1000:.2f}g/kg, "
                f"phi={self.phi*100:.1f}%, h={self.h:.1f}kJ/kg)")


@dataclass
class ComponentResult:
    """Result of a component calculation.

    Attributes:
        air_out: Outlet air state
        Q_heat: Heating energy [kWh] (positive = heating)
        Q_cool: Cooling energy [kWh] (positive = cooling)
        W_el: Electrical energy [kWh]
        m_water: Water consumption [kg] (humidification)
        info: Additional component-specific information
    """
    air_out: AirState
    Q_heat: float = 0.0
    Q_cool: float = 0.0
    W_el: float = 0.0
    m_water: float = 0.0
    info: dict = field(default_factory=dict)

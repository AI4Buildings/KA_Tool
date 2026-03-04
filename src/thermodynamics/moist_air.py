"""Thermodynamic functions for moist air calculations.

Based on the MATLAB functions: ps.m, xs.m, x.m, phi.m, h.m, T.m, T_h_phi.m, Ts.m, Ts2.m
All calculations follow standard psychrometric relationships.

Units:
    T: temperature [°C]
    x: absolute humidity [kg_H2O / kg_dry_air]
    phi: relative humidity [0..1]
    p: pressure [Pa]
    h: specific enthalpy [kJ/kg_dry_air]
"""

import math
from scipy.optimize import brentq

# Thermodynamic constants
C_PL = 1.004    # specific heat dry air [kJ/(kg·K)]
C_PD = 1.86     # specific heat water vapor [kJ/(kg·K)]  (MATLAB uses 1.85)
C_W = 4.18      # specific heat liquid water [kJ/(kg·K)]
C_ICE = 2.04    # specific heat ice [kJ/(kg·K)]
R_0 = 2500.9    # latent heat of vaporization at 0°C [kJ/kg]
SIGMA_0 = 333.1 # latent heat of fusion at 0°C [kJ/kg]

# Simplified constants matching MATLAB implementation
_C_PL_SIMPLE = 1.0   # MATLAB uses 1.0 for c_pl in enthalpy
_C_PD_SIMPLE = 1.85  # MATLAB uses 1.85 for c_pd in enthalpy
_R_0_SIMPLE = 2500.0 # MATLAB uses 2500 for r_0 in enthalpy


def saturation_pressure(T: float) -> float:
    """Saturation vapor pressure of water [Pa].

    Uses Magnus formula with separate coefficients for T < 0°C and T >= 0°C.
    Matches MATLAB ps.m exactly.

    Args:
        T: Temperature [°C]

    Returns:
        Saturation pressure [Pa]
    """
    if T < 0.0:
        return 611.0 * math.exp(22.44 * T / (272.44 + T))
    else:
        return 611.0 * math.exp(17.08 * T / (234.18 + T))


def saturation_humidity(T: float, p: float = 101325.0) -> float:
    """Saturation absolute humidity [kg/kg].

    Matches MATLAB xs.m: x_s = 0.622 * ps(T) / (p - ps(T))

    Args:
        T: Temperature [°C]
        p: Atmospheric pressure [Pa]

    Returns:
        Saturation absolute humidity [kg_H2O/kg_dry_air]
    """
    ps = saturation_pressure(T)
    if p <= ps:
        return float('inf')
    return 0.622 * ps / (p - ps)


def absolute_humidity(phi: float, T: float, p: float = 101325.0) -> float:
    """Absolute humidity from relative humidity and temperature [kg/kg].

    Matches MATLAB x.m: x = 0.622 * phi * ps(T) / (p - phi * ps(T))

    Args:
        phi: Relative humidity [0..1]
        T: Temperature [°C]
        p: Atmospheric pressure [Pa]

    Returns:
        Absolute humidity [kg_H2O/kg_dry_air]
    """
    ps = saturation_pressure(T)
    denom = p - phi * ps
    if denom <= 0:
        return float('inf')
    return 0.622 * phi * ps / denom


def relative_humidity(x: float, T: float, p: float = 101325.0) -> float:
    """Relative humidity from absolute humidity and temperature.

    Matches MATLAB phi.m: phi = min(x * p / (ps(T) * (0.622 + x)), 1)

    Args:
        x: Absolute humidity [kg_H2O/kg_dry_air]
        T: Temperature [°C]
        p: Atmospheric pressure [Pa]

    Returns:
        Relative humidity [0..1]
    """
    ps = saturation_pressure(T)
    if ps == 0:
        return 0.0
    phi = x * p / (ps * (0.622 + x))
    return min(phi, 1.0)


def enthalpy(T: float, x: float) -> float:
    """Specific enthalpy of moist air [kJ/kg_dry_air].

    Simplified formula matching MATLAB h.m: h = 1.0*T + x*(2500 + 1.85*T)
    This is the unsaturated case (most common in HVAC).

    For the full three-case model (as in CLAUDE.md):
    - Case 1: unsaturated (x <= x_s): h = c_pl*T + x*(r_0 + c_pd*T)
    - Case 2: supersaturated T >= 0: h = c_pl*T + x_s*(r_0 + c_pd*T) + (x-x_s)*c_w*T
    - Case 3: supersaturated T < 0: h = c_pl*T + x_s*(r_0 + c_pd*T) + (x-x_s)*(-sigma_0 + c_ice*T)

    Args:
        T: Temperature [°C]
        x: Absolute humidity [kg_H2O/kg_dry_air]

    Returns:
        Specific enthalpy [kJ/kg_dry_air]
    """
    return _C_PL_SIMPLE * T + x * (_R_0_SIMPLE + _C_PD_SIMPLE * T)


def enthalpy_full(T: float, x: float, p: float = 101325.0) -> float:
    """Full specific enthalpy including supersaturation cases [kJ/kg_dry_air].

    Args:
        T: Temperature [°C]
        x: Absolute humidity [kg_H2O/kg_dry_air]
        p: Atmospheric pressure [Pa]

    Returns:
        Specific enthalpy [kJ/kg_dry_air]
    """
    x_s = saturation_humidity(T, p)

    if x <= x_s:
        # Case 1: unsaturated
        return C_PL * T + x * (R_0 + C_PD * T)
    elif T >= 0:
        # Case 2: supersaturated, liquid water
        return C_PL * T + x_s * (R_0 + C_PD * T) + (x - x_s) * C_W * T
    else:
        # Case 3: supersaturated, ice
        return C_PL * T + x_s * (R_0 + C_PD * T) + (x - x_s) * (-SIGMA_0 + C_ICE * T)


def temperature_from_h_x(h: float, x: float) -> float:
    """Temperature from enthalpy and absolute humidity [°C].

    Matches MATLAB T.m: T = (h - 2500*x) / (1 + 1.85*x)

    Args:
        h: Specific enthalpy [kJ/kg_dry_air]
        x: Absolute humidity [kg_H2O/kg_dry_air]

    Returns:
        Temperature [°C]
    """
    return (h - _R_0_SIMPLE * x) / (_C_PL_SIMPLE + _C_PD_SIMPLE * x)


def temperature_from_h_phi(h: float, phi: float, p: float = 101325.0) -> float:
    """Temperature from enthalpy and relative humidity [°C].

    Matches MATLAB T_h_phi.m: uses optimization to find T.

    Args:
        h: Specific enthalpy [kJ/kg_dry_air]
        phi: Relative humidity [0..1]
        p: Atmospheric pressure [Pa]

    Returns:
        Temperature [°C]
    """
    def residual(T):
        x_val = absolute_humidity(phi, T, p)
        return h - enthalpy(T, x_val)

    try:
        return brentq(residual, -40.0, 100.0)
    except ValueError:
        # Fallback: try wider range
        return brentq(residual, -60.0, 150.0)


def dew_point(x: float, p: float = 101325.0) -> float:
    """Dew point temperature [°C].

    Matches MATLAB Ts.m: finds T where x_s(T) = x.

    Args:
        x: Absolute humidity [kg_H2O/kg_dry_air]
        p: Atmospheric pressure [Pa]

    Returns:
        Dew point temperature [°C]
    """
    if x <= 0:
        return -273.15

    def residual(T):
        return x - saturation_humidity(T, p)

    try:
        return brentq(residual, -60.0, 100.0)
    except ValueError:
        # Fallback approximation (MATLAB Ts2.m)
        return dew_point_approx(x)


def dew_point_approx(x: float) -> float:
    """Approximate dew point temperature [°C].

    Matches MATLAB Ts2.m: T_s = 207.8 * x^0.1293 - 100.7

    Args:
        x: Absolute humidity [kg_H2O/kg_dry_air]

    Returns:
        Approximate dew point temperature [°C]
    """
    if x <= 0:
        return -273.15
    return 207.8 * (x ** 0.1293) - 100.7


def heat_flow(m_dot_dry: float, h_out: float, h_in: float) -> float:
    """Heat flow rate for air state change [kW].

    Q_dot = m_dot_dry * (h_out - h_in)

    Args:
        m_dot_dry: Dry air mass flow rate [kg/s]
        h_out: Outlet enthalpy [kJ/kg]
        h_in: Inlet enthalpy [kJ/kg]

    Returns:
        Heat flow rate [kW]
    """
    return m_dot_dry * (h_out - h_in)


def mixing(m_dot_1: float, T_1: float, x_1: float,
           m_dot_2: float, T_2: float, x_2: float,
           ) -> tuple[float, float, float, float]:
    """Adiabatic mixing of two air streams.

    Args:
        m_dot_1: Mass flow rate stream 1 [kg/s]
        T_1: Temperature stream 1 [°C]
        x_1: Absolute humidity stream 1 [kg/kg]
        m_dot_2: Mass flow rate stream 2 [kg/s]
        T_2: Temperature stream 2 [°C]
        x_2: Absolute humidity stream 2 [kg/kg]

    Returns:
        Tuple of (m_dot_3, T_3, x_3, h_3)
    """
    h_1 = enthalpy(T_1, x_1)
    h_2 = enthalpy(T_2, x_2)

    m_dot_3 = m_dot_1 + m_dot_2
    if m_dot_3 == 0:
        return 0.0, 0.0, 0.0, 0.0

    x_3 = (x_1 * m_dot_1 + x_2 * m_dot_2) / m_dot_3
    h_3 = (h_1 * m_dot_1 + h_2 * m_dot_2) / m_dot_3
    T_3 = temperature_from_h_x(h_3, x_3)

    return m_dot_3, T_3, x_3, h_3


def steam_enthalpy(T_steam: float) -> float:
    """Specific enthalpy of steam for humidification [kJ/kg].

    h_steam = r_0 + c_pd * T_steam

    Args:
        T_steam: Steam temperature [°C]

    Returns:
        Specific enthalpy [kJ/kg]
    """
    return _R_0_SIMPLE + _C_PD_SIMPLE * T_steam


def water_enthalpy(T_water: float) -> float:
    """Specific enthalpy of liquid water for spray humidification [kJ/kg].

    h_water = c_w * T_water

    Args:
        T_water: Water temperature [°C]

    Returns:
        Specific enthalpy [kJ/kg]
    """
    return C_W * T_water


def density_moist_air(T: float, x: float, p: float = 101325.0) -> float:
    """Density of moist air [kg/m³].

    Args:
        T: Temperature [°C]
        x: Absolute humidity [kg_H2O/kg_dry_air]
        p: Atmospheric pressure [Pa]

    Returns:
        Density [kg/m³] (of the mixture, i.e. (1+x) * rho_dry)
    """
    T_K = T + 273.15
    R_d = 287.058  # J/(kg·K) specific gas constant dry air
    R_v = 461.5    # J/(kg·K) specific gas constant water vapor

    # Partial pressures
    p_v = x * p / (0.622 + x)
    p_d = p - p_v

    rho = p_d / (R_d * T_K) + p_v / (R_v * T_K)
    return rho


def mass_flow_from_volume_flow(V_dot_m3h: float, T: float, x: float,
                                p: float = 101325.0) -> float:
    """Convert volume flow [m³/h] to dry air mass flow [kg/s].

    Args:
        V_dot_m3h: Volume flow rate [m³/h]
        T: Temperature [°C]
        x: Absolute humidity [kg_H2O/kg_dry_air]
        p: Atmospheric pressure [Pa]

    Returns:
        Dry air mass flow rate [kg/s]
    """
    rho = density_moist_air(T, x, p)
    V_dot_m3s = V_dot_m3h / 3600.0
    m_dot_moist = rho * V_dot_m3s  # kg/s of moist air
    m_dot_dry = m_dot_moist / (1.0 + x)  # kg/s of dry air
    return m_dot_dry

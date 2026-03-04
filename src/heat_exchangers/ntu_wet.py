"""NTU method for wet heat exchangers (cooling coils with dehumidification).

Implements the enthalpy-based NTU-epsilon method for cooling coils operating
below the dew point, where simultaneous heat and mass transfer (condensation)
occurs. Based on the methodology described in the Masterarbeit and Kaup (2015)
part-load correction.

The key difference from dry heat exchangers is that effectiveness is defined
on an enthalpy basis rather than a temperature basis, using the concept of
an effective specific heat capacity c_s that links the air-side enthalpy
change to the water-side temperature change.

Part-load correction exponent: n = 0.2113 (validated against manufacturer
data from Kelvion/iPCoil, mean deviation ~5%).

Units:
    T: temperature [degC]
    x: absolute humidity [kg_H2O / kg_dry_air]
    h: specific enthalpy [kJ/kg_dry_air]
    m_dot: mass flow rate [kg/s]
    V_dot: volume flow rate [m3/h]
    UA_star: overall mass transfer coefficient [kW/(kg/s)] or equivalent
    Q_dot: heat flow rate [kW]
    p: pressure [Pa]
"""

from dataclasses import dataclass
from typing import Optional

import math
from scipy.optimize import brentq

from src.thermodynamics import moist_air

# Constants
C_P_WATER = 4.18       # specific heat of water [kJ/(kg*K)]
RHO_WATER = 1000.0     # density of water [kg/m3]
KAUP_EXPONENT_WET = 0.2113  # part-load exponent for wet HX (Kaup 2015)


def enthalpy_saturated_air(T_water: float, p: float = 101325.0) -> float:
    """Enthalpy of saturated air at a given temperature.

    Computes the specific enthalpy of moist air at 100% relative humidity
    (saturation) at the specified temperature. This represents the enthalpy
    of the air film at the water surface in a wet cooling coil.

    Args:
        T_water: Temperature at which to evaluate saturation [degC].
        p: Atmospheric pressure [Pa].

    Returns:
        Specific enthalpy of saturated air [kJ/kg_dry_air].
    """
    x_sat = moist_air.saturation_humidity(T_water, p)
    return moist_air.enthalpy(T_water, x_sat)


def c_s_effective(T_W_in: float, T_W_out: float, p: float = 101325.0) -> float:
    """Effective specific heat capacity for wet coil analysis.

    Defined as the slope of the saturation enthalpy curve between the water
    inlet and outlet temperatures:

        c_s = (h_sat(T_W_in) - h_sat(T_W_out)) / (T_W_in - T_W_out)

    This linearizes the enthalpy-temperature relationship of saturated air
    over the operating range of the cooling coil.

    Args:
        T_W_in: Water inlet temperature [degC].
        T_W_out: Water outlet temperature [degC].
        p: Atmospheric pressure [Pa].

    Returns:
        Effective specific heat [kJ/(kg_dry_air * K)].

    Raises:
        ValueError: If water inlet and outlet temperatures are equal.
    """
    if abs(T_W_in - T_W_out) < 1e-10:
        raise ValueError(
            "Water inlet and outlet temperatures must differ "
            f"(T_W_in={T_W_in}, T_W_out={T_W_out})."
        )
    h_sat_in = enthalpy_saturated_air(T_W_in, p)
    h_sat_out = enthalpy_saturated_air(T_W_out, p)
    return (h_sat_in - h_sat_out) / (T_W_in - T_W_out)


def _effectiveness_counterflow(NTU: float, C_star: float) -> float:
    """Counterflow heat exchanger effectiveness.

    Args:
        NTU: Number of transfer units [-].
        C_star: Capacity rate ratio [-] (must be in [0, 1]).

    Returns:
        Effectiveness epsilon [-].
    """
    if NTU < 0:
        return 0.0
    if abs(C_star - 1.0) < 1e-10:
        # Special case C* = 1
        return NTU / (1.0 + NTU)
    exp_term = math.exp(-NTU * (1.0 - C_star))
    return (1.0 - exp_term) / (1.0 - C_star * exp_term)


def _invert_effectiveness_counterflow(epsilon: float, C_star: float) -> float:
    """Invert counterflow effectiveness to find NTU from epsilon and C*.

    Uses scipy.optimize.brentq for robust root finding.

    Args:
        epsilon: Effectiveness [-] (must be in (0, 1)).
        C_star: Capacity rate ratio [-].

    Returns:
        NTU: Number of transfer units [-].
    """
    if epsilon <= 0.0:
        return 0.0
    if epsilon >= 1.0:
        return float('inf')

    # For C* = 1: epsilon = NTU/(1+NTU) => NTU = epsilon/(1-epsilon)
    if abs(C_star - 1.0) < 1e-10:
        return epsilon / (1.0 - epsilon)

    def residual(ntu):
        return _effectiveness_counterflow(ntu, C_star) - epsilon

    # Upper bound: for very high effectiveness, NTU can be large
    ntu_max = 50.0
    try:
        return brentq(residual, 0.0, ntu_max, xtol=1e-10)
    except ValueError:
        # If epsilon is very close to theoretical max, return large NTU
        return ntu_max


@dataclass
class WetHeatExchangerDesign:
    """Design (rating) parameters for a wet cooling coil.

    All values refer to the nominal design operating point used to derive
    the reference NTU* and UA* for part-load calculations.

    Attributes:
        capacity_kW: Design cooling capacity [kW] (positive value).
        T_air_in: Design air inlet temperature [degC].
        phi_air_in: Design air inlet relative humidity [0..1].
        T_air_out: Design air outlet temperature [degC].
        phi_air_out: Design air outlet relative humidity [0..1].
        T_water_in: Design water (coolant) inlet temperature [degC].
        T_water_out: Design water (coolant) outlet temperature [degC].
        V_dot_air: Design air volume flow rate [m3/h].
        V_dot_water: Design water volume flow rate [m3/h].
            If not provided, it is calculated from capacity and water temps.
        p: Atmospheric pressure [Pa].
    """
    capacity_kW: float
    T_air_in: float
    phi_air_in: float
    T_air_out: float
    phi_air_out: float
    T_water_in: float
    T_water_out: float
    V_dot_air: float
    V_dot_water: Optional[float] = None
    p: float = 101325.0

    def __post_init__(self):
        """Calculate derived design quantities."""
        # Water volume flow from energy balance if not provided
        if self.V_dot_water is None:
            # Q = m_dot_W * c_p_W * (T_W_out - T_W_in)
            # m_dot_W = Q / (c_p_W * dT)
            dT_water = self.T_water_out - self.T_water_in
            if abs(dT_water) < 1e-10:
                raise ValueError("Design water temperature difference is zero.")
            m_dot_water = self.capacity_kW / (C_P_WATER * dT_water)
            self.V_dot_water = m_dot_water * 3600.0 / RHO_WATER


class WetHeatExchanger:
    """Wet heat exchanger (cooling coil with dehumidification) using NTU method.

    Implements the enthalpy-based NTU-epsilon method with Kaup (2015) part-load
    correction (exponent n=0.2113).

    The calculation procedure:
        1. Determine reference effectiveness epsilon*_ref from design point.
        2. Compute reference m*_ref and invert to find NTU*_ref.
        3. Derive UA*_ref = NTU*_ref * m_dot_L_ref.
        4. For part-load: correct UA* using Kaup formula.
        5. Compute new m*, NTU*, epsilon* at actual operating conditions.
        6. Determine outlet air state (enthalpy, temperature, humidity) and power.

    Attributes:
        design: Design parameters (WetHeatExchangerDesign).
        epsilon_ref: Reference effectiveness at design point [-].
        m_star_ref: Reference modified capacity ratio [-].
        NTU_star_ref: Reference mass transfer NTU [-].
        UA_star_ref: Reference overall transfer coefficient.
        c_s_ref: Reference effective specific heat [kJ/(kg*K)].
        m_dot_air_ref: Reference dry air mass flow [kg/s].
        m_dot_water_ref: Reference water mass flow [kg/s].
    """

    def __init__(self, design: WetHeatExchangerDesign):
        """Initialize from design point data.

        Args:
            design: Design operating point parameters.
        """
        self.design = design
        p = design.p

        # Air states at design point
        x_air_in = moist_air.absolute_humidity(design.phi_air_in, design.T_air_in, p)
        x_air_out = moist_air.absolute_humidity(design.phi_air_out, design.T_air_out, p)
        h_air_in = moist_air.enthalpy(design.T_air_in, x_air_in)
        h_air_out = moist_air.enthalpy(design.T_air_out, x_air_out)

        # Saturated air enthalpy at water inlet temperature
        h_sat_w_in = enthalpy_saturated_air(design.T_water_in, p)

        # Reference effectiveness (enthalpy-based)
        dh_max = h_air_in - h_sat_w_in
        if abs(dh_max) < 1e-10:
            raise ValueError(
                "Design conditions yield zero enthalpy potential. "
                "Check that air inlet enthalpy exceeds saturation enthalpy "
                "at water inlet temperature."
            )
        self.epsilon_ref = (h_air_in - h_air_out) / dh_max

        # Effective specific heat (held constant for part-load)
        self.c_s_ref = c_s_effective(design.T_water_in, design.T_water_out, p)

        # Reference mass flow rates
        # Air: convert volume flow to dry air mass flow at design inlet conditions
        self.m_dot_air_ref = moist_air.mass_flow_from_volume_flow(
            design.V_dot_air, design.T_air_in, x_air_in, p
        )

        # Water mass flow [kg/s]
        self.m_dot_water_ref = design.V_dot_water * RHO_WATER / 3600.0

        # Modified capacity ratio m*
        # m* = (m_dot_L * c_s) / (m_dot_W * c_p_W)
        self.m_star_ref = (
            (self.m_dot_air_ref * self.c_s_ref)
            / (self.m_dot_water_ref * C_P_WATER)
        )

        # Invert effectiveness to get NTU*_ref
        # The counterflow formulas apply with m_star playing the role of C_star
        # Ensure m_star <= 1 for the formula (swap roles if needed)
        if self.m_star_ref <= 1.0:
            self.NTU_star_ref = _invert_effectiveness_counterflow(
                self.epsilon_ref, self.m_star_ref
            )
        else:
            # If m* > 1, effectiveness on air side is limited differently.
            # epsilon_water = epsilon_air * m_star
            # Use water-side effectiveness for inversion with 1/m*
            eps_water = self.epsilon_ref * self.m_star_ref
            if eps_water > 1.0:
                eps_water = min(eps_water, 0.9999)
            ntu_water = _invert_effectiveness_counterflow(
                eps_water, 1.0 / self.m_star_ref
            )
            # NTU* (air-side) = NTU_water * m_star
            self.NTU_star_ref = ntu_water * self.m_star_ref

        # UA*_ref = NTU*_ref * m_dot_L_ref
        self.UA_star_ref = self.NTU_star_ref * self.m_dot_air_ref

        # Store design volume flows for Kaup correction
        self._V_dot_air_ref = design.V_dot_air
        self._V_dot_water_ref = design.V_dot_water

    def calculate(
        self,
        T_air_in: float,
        x_air_in: float,
        V_dot_air: float,
        T_water_in: float,
        V_dot_water: Optional[float] = None,
        p: float = 101325.0,
    ) -> dict:
        """Calculate outlet conditions and cooling power at given operating point.

        Implements the full part-load procedure:
            1. Kaup correction for UA* at actual volume flows.
            2. New m*, NTU*, epsilon* at actual conditions.
            3. Outlet air enthalpy from effectiveness.
            4. Outlet air temperature and humidity (assuming saturation at coil).
            5. Water outlet temperature from energy balance.
            6. Cooling power.

        Args:
            T_air_in: Inlet air temperature [degC].
            x_air_in: Inlet air absolute humidity [kg_H2O/kg_dry_air].
            V_dot_air: Actual air volume flow rate [m3/h].
            T_water_in: Actual water inlet temperature [degC].
            V_dot_water: Actual water volume flow rate [m3/h].
                If None, uses design value.
            p: Atmospheric pressure [Pa].

        Returns:
            Dictionary with keys:
                T_air_out: Outlet air temperature [degC].
                x_air_out: Outlet air absolute humidity [kg/kg].
                h_air_out: Outlet air enthalpy [kJ/kg].
                phi_air_out: Outlet air relative humidity [0..1].
                T_water_out: Outlet water temperature [degC].
                Q_dot: Cooling power [kW] (positive = cooling delivered).
                epsilon_star: Actual effectiveness [-].
                NTU_star: Actual NTU* [-].
                m_star: Actual modified capacity ratio [-].
                UA_star: Actual overall transfer coefficient.
                condensate_kg_per_s: Condensate mass flow [kg/s].
        """
        if V_dot_water is None:
            V_dot_water = self._V_dot_water_ref

        # --- Step 1: Air and water mass flows ---
        m_dot_air = moist_air.mass_flow_from_volume_flow(
            V_dot_air, T_air_in, x_air_in, p
        )
        m_dot_water = V_dot_water * RHO_WATER / 3600.0

        # --- Step 2: Check if cooling is possible ---
        h_air_in = moist_air.enthalpy(T_air_in, x_air_in)
        h_sat_w_in = enthalpy_saturated_air(T_water_in, p)

        if h_air_in <= h_sat_w_in:
            # No cooling potential: air enthalpy is already at or below
            # saturation enthalpy at water inlet temperature.
            return {
                "T_air_out": T_air_in,
                "x_air_out": x_air_in,
                "h_air_out": h_air_in,
                "phi_air_out": moist_air.relative_humidity(x_air_in, T_air_in, p),
                "T_water_out": T_water_in,
                "Q_dot": 0.0,
                "epsilon_star": 0.0,
                "NTU_star": 0.0,
                "m_star": 0.0,
                "UA_star": 0.0,
                "condensate_kg_per_s": 0.0,
            }

        # --- Step 3: Kaup part-load correction for UA* ---
        V_ratio_air = V_dot_air / self._V_dot_air_ref
        V_ratio_water = V_dot_water / self._V_dot_water_ref
        UA_star = self.UA_star_ref * (V_ratio_air * V_ratio_water) ** KAUP_EXPONENT_WET

        # --- Step 4: New m* and NTU* ---
        # c_s is held constant at design value (linearization assumption)
        m_star = (m_dot_air * self.c_s_ref) / (m_dot_water * C_P_WATER)
        NTU_star = UA_star / m_dot_air if m_dot_air > 0 else 0.0

        # --- Step 5: New effectiveness ---
        if m_star <= 1.0:
            epsilon_star = _effectiveness_counterflow(NTU_star, m_star)
        else:
            # Air-side is the larger capacity stream.
            # Compute water-side NTU and effectiveness, then convert.
            NTU_water = NTU_star / m_star if m_star > 0 else 0.0
            eps_water = _effectiveness_counterflow(NTU_water, 1.0 / m_star)
            epsilon_star = eps_water / m_star if m_star > 0 else 0.0

        # --- Step 6: Outlet air enthalpy ---
        h_air_out = h_air_in - epsilon_star * (h_air_in - h_sat_w_in)

        # --- Step 7: Outlet air temperature and humidity ---
        # The air leaving a wet coil is approximately at saturation (phi ~ 0.90-1.0).
        # Find temperature such that enthalpy of saturated air equals h_air_out.
        # h_sat(T) = h_air_out => solve for T
        T_air_out = _temperature_from_saturated_enthalpy(h_air_out, p)

        # At the coil outlet, the air is essentially saturated
        x_air_out = moist_air.saturation_humidity(T_air_out, p)

        # If the computed outlet humidity is higher than inlet (shouldn't happen
        # in cooling), cap it at inlet value
        if x_air_out > x_air_in:
            x_air_out = x_air_in
            T_air_out = moist_air.temperature_from_h_x(h_air_out, x_air_out)

        phi_air_out = moist_air.relative_humidity(x_air_out, T_air_out, p)

        # --- Step 8: Cooling power ---
        Q_dot = m_dot_air * (h_air_in - h_air_out)  # [kW]

        # --- Step 9: Water outlet temperature from energy balance ---
        if m_dot_water > 0 and C_P_WATER > 0:
            T_water_out = T_water_in + Q_dot / (m_dot_water * C_P_WATER)
        else:
            T_water_out = T_water_in

        # --- Step 10: Condensate ---
        condensate = max(0.0, m_dot_air * (x_air_in - x_air_out))

        return {
            "T_air_out": T_air_out,
            "x_air_out": x_air_out,
            "h_air_out": h_air_out,
            "phi_air_out": phi_air_out,
            "T_water_out": T_water_out,
            "Q_dot": Q_dot,
            "epsilon_star": epsilon_star,
            "NTU_star": NTU_star,
            "m_star": m_star,
            "UA_star": UA_star,
            "condensate_kg_per_s": condensate,
        }

    def calculate_target_outlet(
        self,
        T_air_in: float,
        x_air_in: float,
        V_dot_air: float,
        T_water_in: float,
        T_air_target: float,
        x_air_target: Optional[float] = None,
        p: float = 101325.0,
    ) -> dict:
        """Calculate required water flow to achieve a target outlet condition.

        Iterates over water volume flow to find the operating point where
        the outlet air temperature (or humidity) matches the target.

        Args:
            T_air_in: Inlet air temperature [degC].
            x_air_in: Inlet air absolute humidity [kg/kg].
            V_dot_air: Actual air volume flow [m3/h].
            T_water_in: Water inlet temperature [degC].
            T_air_target: Target outlet air temperature [degC].
            x_air_target: Target outlet absolute humidity [kg/kg] (optional).
                If provided, the iteration targets humidity instead of temperature.
            p: Atmospheric pressure [Pa].

        Returns:
            Same dictionary as calculate(), plus:
                V_dot_water_required: Required water volume flow [m3/h].
        """
        # Determine what we are targeting
        if x_air_target is not None:
            def objective(V_dot_w):
                result = self.calculate(
                    T_air_in, x_air_in, V_dot_air, T_water_in, V_dot_w, p
                )
                return result["x_air_out"] - x_air_target
        else:
            def objective(V_dot_w):
                result = self.calculate(
                    T_air_in, x_air_in, V_dot_air, T_water_in, V_dot_w, p
                )
                return result["T_air_out"] - T_air_target

        # Search range: from very small flow to large flow
        V_dot_min = self._V_dot_water_ref * 0.05
        V_dot_max = self._V_dot_water_ref * 3.0

        try:
            V_dot_water_req = brentq(objective, V_dot_min, V_dot_max, xtol=0.01)
        except ValueError:
            # Target may not be reachable; return result at max water flow
            V_dot_water_req = V_dot_max

        result = self.calculate(
            T_air_in, x_air_in, V_dot_air, T_water_in, V_dot_water_req, p
        )
        result["V_dot_water_required"] = V_dot_water_req
        return result


def _temperature_from_saturated_enthalpy(h_target: float, p: float = 101325.0) -> float:
    """Find temperature where enthalpy of saturated air equals h_target.

    Solves h_sat(T) = h_target for T, where h_sat(T) = enthalpy(T, x_sat(T)).

    Args:
        h_target: Target enthalpy [kJ/kg_dry_air].
        p: Atmospheric pressure [Pa].

    Returns:
        Temperature [degC].
    """
    def residual(T):
        return enthalpy_saturated_air(T, p) - h_target

    try:
        return brentq(residual, -20.0, 60.0, xtol=1e-6)
    except ValueError:
        # Widen search range as fallback
        return brentq(residual, -40.0, 80.0, xtol=1e-6)

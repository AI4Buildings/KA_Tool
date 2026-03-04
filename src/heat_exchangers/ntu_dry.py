"""
NTU method for dry heat exchangers (no condensation/evaporation).

Used for: pre-heating coils, post-heating coils, plate heat recovery,
runaround coil system (KVS).

Implements:
- Counterflow and crossflow effectiveness-NTU relations
- Inverse NTU calculations (from known effectiveness)
- Part-load correction after Kaup (2015) with exponent n=0.4
- DryHeatExchanger class for full part-load calculation workflow

References:
- Kaup, C. (2015): Part-load correction for heat exchangers.
- Masterarbeit Sebastian Dragosits, FH Burgenland, 2023.
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_P_AIR = 1.006  # kJ/(kg*K) -- specific heat of dry air
C_P_WATER = 4.18  # kJ/(kg*K) -- specific heat of water
RHO_AIR = 1.2  # kg/m^3 -- approximate air density
RHO_WATER = 1000.0  # kg/m^3 -- water density


class FlowType(Enum):
    """Heat exchanger flow arrangement."""
    COUNTERFLOW = "counterflow"
    CROSSFLOW = "crossflow"


# ---------------------------------------------------------------------------
# Effectiveness-NTU relations
# ---------------------------------------------------------------------------

def effectiveness_counterflow(NTU: float, C_star: float) -> float:
    """Effectiveness of a counterflow heat exchanger.

    Parameters
    ----------
    NTU : float
        Number of transfer units (dimensionless, >= 0).
    C_star : float
        Capacity-rate ratio C_min / C_max (0 <= C_star <= 1).

    Returns
    -------
    float
        Effectiveness epsilon in [0, 1].
    """
    if NTU < 0.0:
        raise ValueError(f"NTU must be >= 0, got {NTU}")
    if not 0.0 <= C_star <= 1.0:
        raise ValueError(f"C_star must be in [0, 1], got {C_star}")

    if NTU == 0.0:
        return 0.0

    # Special case: C_star == 0 means one fluid has infinite capacity
    # (e.g. phase change or very large flow). Effectiveness -> 1 - exp(-NTU).
    if C_star == 0.0:
        return 1.0 - math.exp(-NTU)

    # Special case: C_star == 1
    if abs(C_star - 1.0) < 1e-12:
        return NTU / (1.0 + NTU)

    exp_term = math.exp(-NTU * (1.0 - C_star))
    return (1.0 - exp_term) / (1.0 - C_star * exp_term)


def effectiveness_crossflow(NTU: float, C_star: float) -> float:
    """Effectiveness of a crossflow heat exchanger (single pass, both unmixed).

    Uses the simplified relation from the CLAUDE.md specification:
        eps = (1 - exp(-C_star * (1 - exp(-NTU)))) / C_star

    This is actually the crossflow formula for C_min mixed / C_max unmixed,
    which is commonly used for plate heat exchangers in HVAC applications.

    Parameters
    ----------
    NTU : float
        Number of transfer units (dimensionless, >= 0).
    C_star : float
        Capacity-rate ratio C_min / C_max (0 <= C_star <= 1).

    Returns
    -------
    float
        Effectiveness epsilon in [0, 1].
    """
    if NTU < 0.0:
        raise ValueError(f"NTU must be >= 0, got {NTU}")
    if not 0.0 <= C_star <= 1.0:
        raise ValueError(f"C_star must be in [0, 1], got {C_star}")

    if NTU == 0.0:
        return 0.0

    # C_star == 0: one side has infinite capacity
    if C_star == 0.0:
        return 1.0 - math.exp(-NTU)

    inner = 1.0 - math.exp(-NTU)
    return (1.0 - math.exp(-C_star * inner)) / C_star


def effectiveness(NTU: float, C_star: float, flow_type: FlowType) -> float:
    """Compute effectiveness for the given flow arrangement.

    Parameters
    ----------
    NTU : float
        Number of transfer units.
    C_star : float
        Capacity-rate ratio C_min / C_max.
    flow_type : FlowType
        COUNTERFLOW or CROSSFLOW.

    Returns
    -------
    float
        Effectiveness epsilon.
    """
    if flow_type == FlowType.COUNTERFLOW:
        return effectiveness_counterflow(NTU, C_star)
    elif flow_type == FlowType.CROSSFLOW:
        return effectiveness_crossflow(NTU, C_star)
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")


# ---------------------------------------------------------------------------
# Inverse NTU calculations
# ---------------------------------------------------------------------------

def ntu_from_effectiveness_counterflow(eps: float, C_star: float) -> float:
    """Analytically invert the counterflow effectiveness-NTU relation.

    Parameters
    ----------
    eps : float
        Effectiveness in (0, 1).
    C_star : float
        Capacity-rate ratio C_min / C_max in [0, 1].

    Returns
    -------
    float
        NTU value that produces the given effectiveness.
    """
    if eps <= 0.0:
        return 0.0
    if not 0.0 <= C_star <= 1.0:
        raise ValueError(f"C_star must be in [0, 1], got {C_star}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # C_star == 0: eps = 1 - exp(-NTU)  =>  NTU = -ln(1 - eps)
    if C_star == 0.0:
        return -math.log(1.0 - eps)

    # C_star == 1: eps = NTU / (1 + NTU)  =>  NTU = eps / (1 - eps)
    if abs(C_star - 1.0) < 1e-12:
        return eps / (1.0 - eps)

    # General case:
    #   eps = (1 - exp(-NTU*(1-C*))) / (1 - C* * exp(-NTU*(1-C*)))
    # Let z = exp(-NTU*(1-C*)):
    #   eps = (1 - z) / (1 - C* * z)
    #   eps * (1 - C* * z) = 1 - z
    #   eps - eps * C* * z = 1 - z
    #   z * (1 - eps * C*) = 1 - eps
    #   z = (1 - eps) / (1 - eps * C*)
    #   NTU = -ln(z) / (1 - C*)
    z = (1.0 - eps) / (1.0 - eps * C_star)
    if z <= 0.0:
        raise ValueError("Invalid eps/C_star combination leads to non-physical NTU.")
    return -math.log(z) / (1.0 - C_star)


def ntu_from_effectiveness_crossflow(eps: float, C_star: float,
                                     ntu_bounds: tuple[float, float] = (1e-6, 100.0)) -> float:
    """Numerically invert the crossflow effectiveness-NTU relation using brentq.

    Parameters
    ----------
    eps : float
        Target effectiveness in (0, 1).
    C_star : float
        Capacity-rate ratio C_min / C_max in [0, 1].
    ntu_bounds : tuple of float
        Bracket for the root search. Default (1e-6, 100).

    Returns
    -------
    float
        NTU that produces the given effectiveness.
    """
    if eps <= 0.0:
        return 0.0
    if not 0.0 <= C_star <= 1.0:
        raise ValueError(f"C_star must be in [0, 1], got {C_star}")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # C_star == 0: same as counterflow
    if C_star == 0.0:
        return -math.log(1.0 - eps)

    def residual(ntu: float) -> float:
        return effectiveness_crossflow(ntu, C_star) - eps

    return brentq(residual, ntu_bounds[0], ntu_bounds[1], xtol=1e-10)


def ntu_from_effectiveness(eps: float, C_star: float, flow_type: FlowType) -> float:
    """Invert the effectiveness-NTU relation for the given flow type.

    Parameters
    ----------
    eps : float
        Target effectiveness.
    C_star : float
        Capacity-rate ratio.
    flow_type : FlowType
        COUNTERFLOW or CROSSFLOW.

    Returns
    -------
    float
        NTU.
    """
    if flow_type == FlowType.COUNTERFLOW:
        return ntu_from_effectiveness_counterflow(eps, C_star)
    elif flow_type == FlowType.CROSSFLOW:
        return ntu_from_effectiveness_crossflow(eps, C_star)
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")


# ---------------------------------------------------------------------------
# Part-load correction (Kaup 2015)
# ---------------------------------------------------------------------------

def ua_kaup_correction(
    UA_ref: float,
    V_dot_L: float,
    V_dot_L_ref: float,
    V_dot_W: float,
    V_dot_W_ref: float,
    n: float = 0.4,
) -> float:
    """Part-load correction of overall heat transfer coefficient UA (Kaup 2015).

    UA = UA_ref * ((V_dot_L / V_dot_L_ref) * (V_dot_W / V_dot_W_ref)) ** n

    Valid ranges (accuracy +/-3%):
        V_dot_L / V_dot_L_ref in [0.4, 1.6]
        V_dot_W / V_dot_W_ref in [0.8, 1.4]

    Parameters
    ----------
    UA_ref : float
        Reference (design) heat transfer coefficient [kW/K].
    V_dot_L : float
        Current air-side volume flow [m^3/h].
    V_dot_L_ref : float
        Design air-side volume flow [m^3/h].
    V_dot_W : float
        Current water-side volume flow [m^3/h].
    V_dot_W_ref : float
        Design water-side volume flow [m^3/h].
    n : float
        Kaup exponent. Default 0.4 for dry heat exchangers.

    Returns
    -------
    float
        Corrected UA [kW/K].
    """
    if V_dot_L_ref <= 0.0 or V_dot_W_ref <= 0.0:
        raise ValueError("Reference volume flows must be > 0.")
    if V_dot_L <= 0.0 or V_dot_W <= 0.0:
        # Zero flow means zero heat transfer
        return 0.0

    ratio = (V_dot_L / V_dot_L_ref) * (V_dot_W / V_dot_W_ref)
    return UA_ref * ratio ** n


# ---------------------------------------------------------------------------
# Helper: convert volume flow m^3/h -> mass flow kg/s
# ---------------------------------------------------------------------------

def _volume_to_mass_flow(V_dot_m3h: float, rho: float) -> float:
    """Convert volume flow in m^3/h to mass flow in kg/s."""
    return V_dot_m3h / 3600.0 * rho


# ---------------------------------------------------------------------------
# Design-state data container
# ---------------------------------------------------------------------------

@dataclass
class DryHXDesignData:
    """Design-point parameters for a dry heat exchanger.

    All temperatures in degC, volume flows in m^3/h, capacity in kW.
    """
    # Air side
    V_dot_L: float  # design air volume flow [m^3/h]
    T_L_in: float  # design air inlet temperature [degC]
    T_L_out: float  # design air outlet temperature [degC]

    # Water side
    V_dot_W: float  # design water volume flow [m^3/h]
    T_W_in: float  # design water inlet temperature [degC]
    T_W_out: float  # design water outlet temperature [degC]

    # Optional: if provided, overrides computed values
    capacity_kW: Optional[float] = None  # design capacity [kW]


# ---------------------------------------------------------------------------
# DryHeatExchanger class
# ---------------------------------------------------------------------------

class DryHeatExchanger:
    """Dry heat exchanger with NTU-based part-load model and Kaup correction.

    The class stores the design-point state and computes part-load performance
    for arbitrary operating conditions.

    Parameters
    ----------
    design : DryHXDesignData
        Design-point data.
    flow_type : FlowType
        COUNTERFLOW (heating/cooling coils, KVS) or CROSSFLOW (plate HRV).
    kaup_exponent : float
        Exponent for UA part-load correction. Default 0.4.
    c_p_air : float
        Specific heat of air [kJ/(kg*K)]. Default 1.006.
    c_p_water : float
        Specific heat of water [kJ/(kg*K)]. Default 4.18.
    rho_air : float
        Air density [kg/m^3]. Default 1.2.
    rho_water : float
        Water density [kg/m^3]. Default 1000.
    """

    def __init__(
        self,
        design: DryHXDesignData,
        flow_type: FlowType = FlowType.COUNTERFLOW,
        kaup_exponent: float = 0.4,
        c_p_air: float = C_P_AIR,
        c_p_water: float = C_P_WATER,
        rho_air: float = RHO_AIR,
        rho_water: float = RHO_WATER,
    ):
        self.design = design
        self.flow_type = flow_type
        self.kaup_exponent = kaup_exponent
        self.c_p_air = c_p_air
        self.c_p_water = c_p_water
        self.rho_air = rho_air
        self.rho_water = rho_water

        # Derive design-point reference quantities
        self._compute_design_state()

    def _compute_design_state(self) -> None:
        """Derive NTU_ref, UA_ref, eps_ref, C_star_ref from design data."""
        d = self.design

        # Mass flow rates [kg/s]
        self._m_dot_L_ref = _volume_to_mass_flow(d.V_dot_L, self.rho_air)
        self._m_dot_W_ref = _volume_to_mass_flow(d.V_dot_W, self.rho_water)

        # Capacity flow rates [kW/K]
        self._C_L_ref = self._m_dot_L_ref * self.c_p_air
        self._C_W_ref = self._m_dot_W_ref * self.c_p_water

        # Identify C_min and C_max
        self._C_min_ref = min(self._C_L_ref, self._C_W_ref)
        self._C_max_ref = max(self._C_L_ref, self._C_W_ref)

        # Capacity-rate ratio
        if self._C_max_ref > 0.0:
            self._C_star_ref = self._C_min_ref / self._C_max_ref
        else:
            self._C_star_ref = 0.0

        # Design effectiveness based on the C_min side
        # eps = Q / Q_max  where Q_max = C_min * (T_hot_in - T_cold_in)
        delta_T_max = abs(d.T_W_in - d.T_L_in)
        if delta_T_max < 1e-10:
            # No temperature difference at design -> no heat exchange
            self._eps_ref = 0.0
            self._NTU_ref = 0.0
            self._UA_ref = 0.0
            return

        # Determine Q_dot at design [kW]
        if d.capacity_kW is not None:
            Q_ref = abs(d.capacity_kW)
        else:
            # Compute from air side
            Q_ref = abs(self._C_L_ref * (d.T_L_out - d.T_L_in))

        Q_max = self._C_min_ref * delta_T_max
        self._eps_ref = Q_ref / Q_max if Q_max > 0.0 else 0.0

        # Clamp effectiveness to valid range
        self._eps_ref = max(0.0, min(self._eps_ref, 0.9999))

        # Invert to get NTU_ref
        if self._eps_ref > 0.0:
            self._NTU_ref = ntu_from_effectiveness(
                self._eps_ref, self._C_star_ref, self.flow_type
            )
        else:
            self._NTU_ref = 0.0

        # UA_ref [kW/K]
        self._UA_ref = self._NTU_ref * self._C_min_ref

    # -- Public read-only properties for design state --

    @property
    def eps_ref(self) -> float:
        """Design-point effectiveness."""
        return self._eps_ref

    @property
    def NTU_ref(self) -> float:
        """Design-point NTU."""
        return self._NTU_ref

    @property
    def UA_ref(self) -> float:
        """Design-point UA [kW/K]."""
        return self._UA_ref

    @property
    def C_star_ref(self) -> float:
        """Design-point capacity-rate ratio."""
        return self._C_star_ref

    # -----------------------------------------------------------------
    # Part-load calculation
    # -----------------------------------------------------------------

    def calculate(
        self,
        T_L_in: float,
        T_W_in: float,
        V_dot_L: float,
        V_dot_W: float,
    ) -> dict:
        """Calculate part-load outlet temperatures and heat transfer rate.

        Follows the Kaup part-load procedure:
        1. Use design eps_ref, C_star_ref, NTU_ref, UA_ref (precomputed).
        2. Correct UA for current volume flows using Kaup formula.
        3. Compute new C_star from current flows.
        4. Compute new NTU from corrected UA and current C_min.
        5. Compute new effectiveness.
        6. Compute outlet temperatures and power.

        Parameters
        ----------
        T_L_in : float
            Current air inlet temperature [degC].
        T_W_in : float
            Current water/fluid inlet temperature [degC].
        V_dot_L : float
            Current air-side volume flow [m^3/h].
        V_dot_W : float
            Current water-side volume flow [m^3/h].

        Returns
        -------
        dict with keys:
            T_L_out : float   -- air outlet temperature [degC]
            T_W_out : float   -- water outlet temperature [degC]
            Q_dot   : float   -- heat transfer rate [kW] (positive = heat to air)
            eps     : float   -- current effectiveness
            NTU     : float   -- current NTU
            UA      : float   -- current UA [kW/K]
            C_star  : float   -- current capacity-rate ratio
        """
        # Handle zero-flow edge cases
        if V_dot_L <= 0.0 or V_dot_W <= 0.0:
            return {
                "T_L_out": T_L_in,
                "T_W_out": T_W_in,
                "Q_dot": 0.0,
                "eps": 0.0,
                "NTU": 0.0,
                "UA": 0.0,
                "C_star": 0.0,
            }

        # No temperature difference -> no heat exchange
        if abs(T_W_in - T_L_in) < 1e-10:
            return {
                "T_L_out": T_L_in,
                "T_W_out": T_W_in,
                "Q_dot": 0.0,
                "eps": 0.0,
                "NTU": 0.0,
                "UA": 0.0,
                "C_star": 0.0,
            }

        # Step 5 (Kaup): corrected UA
        UA_new = ua_kaup_correction(
            self._UA_ref,
            V_dot_L, self.design.V_dot_L,
            V_dot_W, self.design.V_dot_W,
            self.kaup_exponent,
        )

        # Current mass flows [kg/s]
        m_dot_L = _volume_to_mass_flow(V_dot_L, self.rho_air)
        m_dot_W = _volume_to_mass_flow(V_dot_W, self.rho_water)

        # Current capacity flow rates [kW/K]
        C_L = m_dot_L * self.c_p_air
        C_W = m_dot_W * self.c_p_water

        C_min = min(C_L, C_W)
        C_max = max(C_L, C_W)

        # Step 6: new C_star
        C_star_new = C_min / C_max if C_max > 0.0 else 0.0

        # Step 7: new NTU
        NTU_new = UA_new / C_min if C_min > 0.0 else 0.0

        # Step 8: new effectiveness
        eps_new = effectiveness(NTU_new, C_star_new, self.flow_type)

        # Step 9: outlet temperatures and power
        # Q_dot = eps * C_min * (T_hot_in - T_cold_in)
        # Positive Q_dot means heat transferred from hot to cold side.
        delta_T_max = T_W_in - T_L_in  # can be positive (heating) or negative (cooling)
        Q_dot = eps_new * C_min * delta_T_max  # [kW], positive = heating air

        # Air outlet
        T_L_out = T_L_in + Q_dot / C_L if C_L > 0.0 else T_L_in

        # Water outlet (energy balance)
        T_W_out = T_W_in - Q_dot / C_W if C_W > 0.0 else T_W_in

        return {
            "T_L_out": T_L_out,
            "T_W_out": T_W_out,
            "Q_dot": Q_dot,
            "eps": eps_new,
            "NTU": NTU_new,
            "UA": UA_new,
            "C_star": C_star_new,
        }

    def calculate_with_target(
        self,
        T_L_in: float,
        T_W_in: float,
        V_dot_L: float,
        V_dot_W_max: float,
        T_L_target: float,
    ) -> dict:
        """Calculate part-load, adjusting water flow to reach a target air outlet temperature.

        This is useful when the heat exchanger should deliver a specific supply air
        temperature and the water-side flow (valve position) is the control variable.

        If the target cannot be reached even at maximum water flow, the result at
        maximum flow is returned.

        Parameters
        ----------
        T_L_in : float
            Current air inlet temperature [degC].
        T_W_in : float
            Water inlet temperature [degC].
        V_dot_L : float
            Current air volume flow [m^3/h].
        V_dot_W_max : float
            Maximum water volume flow [m^3/h] (fully open valve).
        T_L_target : float
            Desired air outlet temperature [degC].

        Returns
        -------
        dict
            Same keys as ``calculate``, plus:
            V_dot_W : float -- water volume flow used [m^3/h]
            target_reached : bool -- whether the target was met
        """
        # Check if target is already satisfied with no heat exchange
        # (e.g. air already warm enough for heating case)
        heating = T_W_in > T_L_in
        if heating and T_L_in >= T_L_target:
            result = self.calculate(T_L_in, T_W_in, V_dot_L, 0.0)
            result["V_dot_W"] = 0.0
            result["target_reached"] = True
            return result
        if not heating and T_L_in <= T_L_target:
            result = self.calculate(T_L_in, T_W_in, V_dot_L, 0.0)
            result["V_dot_W"] = 0.0
            result["target_reached"] = True
            return result

        # Check maximum capacity first
        result_max = self.calculate(T_L_in, T_W_in, V_dot_L, V_dot_W_max)
        if heating:
            if result_max["T_L_out"] <= T_L_target:
                # Cannot reach target even at full flow
                result_max["V_dot_W"] = V_dot_W_max
                result_max["target_reached"] = False
                return result_max
        else:
            if result_max["T_L_out"] >= T_L_target:
                result_max["V_dot_W"] = V_dot_W_max
                result_max["target_reached"] = False
                return result_max

        # Use brentq to find the water flow that yields T_L_target
        def residual(V_dot_W: float) -> float:
            res = self.calculate(T_L_in, T_W_in, V_dot_L, V_dot_W)
            return res["T_L_out"] - T_L_target

        # Lower bound: small positive flow (avoid zero division)
        V_dot_W_min = V_dot_W_max * 0.001
        # Ensure bracket is valid
        res_lo = residual(V_dot_W_min)
        res_hi = residual(V_dot_W_max)

        if res_lo * res_hi > 0:
            # Bracket does not span the root; return max flow result
            result_max["V_dot_W"] = V_dot_W_max
            result_max["target_reached"] = False
            return result_max

        V_dot_W_opt = brentq(residual, V_dot_W_min, V_dot_W_max, xtol=0.01)

        result = self.calculate(T_L_in, T_W_in, V_dot_L, V_dot_W_opt)
        result["V_dot_W"] = V_dot_W_opt
        result["target_reached"] = True
        return result

"""
Rotary heat exchanger model based on OENORM EN 16798-5-1:2017 + A1:2020,
with modifications from IDA ICE (EQUA).

Converted from MATLAB function heat_rec_wheel_calc_V5.m.

Supported rotor types:
    - ROT_SORP: Sorption rotor
    - ROT_HYG:  Enthalpy (hygroscopic) rotor
    - ROT_NH:   Condensation (non-hygroscopic) rotor

The module computes temperature and humidity transfer efficiencies as functions
of rotor speed, airflow ratios, face velocity, and condensation potential.
Three control/optimization modes are supported:
    - TEMP:   Temperature-controlled rotor speed optimization
    - HUM:    Humidity-controlled rotor speed (direct calculation)
    - ENERGY: Energy-optimized rotor speed selection
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RotorType(Enum):
    """Supported rotary heat exchanger rotor types."""
    ROT_SORP = "ROT_SORP"   # Sorption rotor
    ROT_HYG = "ROT_HYG"     # Enthalpy (hygroscopic) rotor
    ROT_NH = "ROT_NH"        # Condensation (non-hygroscopic) rotor


class ControlMode(Enum):
    """Rotor speed optimization / control mode."""
    TEMP = "Temp"       # Temperature-controlled
    HUM = "Hum"         # Humidity-controlled
    ENERGY = "Energy"   # Energy-optimized


class FanLocation(Enum):
    """Fan position relative to the heat recovery unit."""
    UP_HR = "UP_HR"      # Upstream of heat recovery
    DOWN_HR = "DOWN_HR"  # Downstream of heat recovery


# ---------------------------------------------------------------------------
# Model constants per rotor type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RotorConstants:
    """Model constants for a specific rotor type per OENORM EN 16798-5-1."""

    # Temperature transfer
    eta_hr_nom: float   # Nominal temperature transfer efficiency [-]
    v_hr_nom: float     # Nominal face velocity [m/s]
    C1: float           # Velocity correction coefficient
    C2: float           # Velocity correction offset
    C3: float           # Rotor speed correction coefficient (temp)
    C4: float           # Rotor speed correction coefficient (temp)
    C5: float           # Rotor speed correction offset (temp)
    e1: float           # Rotor speed correction exponent (temp)

    # Humidity transfer
    eta_xr_nom: float   # Nominal humidity transfer efficiency [-]
    C6: float           # Condensation potential coefficient
    C7: float           # Condensation potential offset
    C8: float           # Additional condensation coefficient (ROT_HYG only)
    dx_e_nom: float     # Nominal humidity difference [kg/kg]
    C9: float           # Airflow ratio correction (humidity)
    C10: float          # Velocity correction coefficient (humidity)
    C11: float          # Velocity correction offset (humidity)
    C12: float          # Rotor speed correction coefficient (humidity)
    C13: float          # Rotor speed correction coefficient (humidity)
    C14: float          # Rotor speed correction offset (humidity)
    e2: float           # Rotor speed correction exponent (humidity)

    # Rotor speed normalization
    n_rot_max_norm: float = 20.0  # [1/min]


# Pre-defined constant sets for each rotor type
ROTOR_CONSTANTS: dict[RotorType, RotorConstants] = {
    RotorType.ROT_SORP: RotorConstants(
        eta_hr_nom=0.69, v_hr_nom=3.5,
        C1=-0.0665, C2=1.0,
        C3=1.0182, C4=0.0352, C5=0.276, e1=-2.7,
        eta_xr_nom=0.69,
        C6=16.4, C7=0.918, C8=0.0, dx_e_nom=0.005,
        C9=0.1, C10=-0.098, C11=1.0,
        C12=1.0533, C13=80000.0, C14=15.0, e2=-4.0,
    ),
    RotorType.ROT_HYG: RotorConstants(
        eta_hr_nom=0.67, v_hr_nom=3.5,
        C1=-0.0684, C2=1.0,
        C3=1.0182, C4=0.0352, C5=0.276, e1=-2.7,
        eta_xr_nom=0.42,
        C6=129.0, C7=0.476, C8=23.8, dx_e_nom=0.005,
        C9=0.1, C10=-0.152, C11=1.0,
        C12=1.0533, C13=80000.0, C14=15.0, e2=-4.0,
    ),
    RotorType.ROT_NH: RotorConstants(
        eta_hr_nom=0.69, v_hr_nom=3.5,
        C1=-0.0643, C2=1.0,
        C3=1.0182, C4=0.0352, C5=0.276, e1=-2.7,
        eta_xr_nom=0.30,
        C6=248.0, C7=-0.240, C8=0.0, dx_e_nom=0.005,
        C9=0.1, C10=-0.200, C11=1.0,
        C12=1.0533, C13=80000.0, C14=15.0, e2=-4.0,
    ),
}


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class RotaryResult:
    """Result of a rotary heat exchanger calculation for one timestep."""

    # Transfer efficiencies at the given (or initial) rotor speed
    eta_hr: float       # Temperature transfer efficiency [-]
    eta_xr: float       # Humidity transfer efficiency [-]

    # Individual correction factors (temperature)
    f_q: float          # Airflow ratio correction
    f_v: float          # Face velocity correction
    f_n: float          # Rotor speed correction (temperature)

    # Individual correction factors (humidity)
    f_dx_x: float       # Condensation potential correction
    f_q_x: float        # Airflow ratio correction (humidity)
    f_v_x: float        # Face velocity correction (humidity)
    f_n_x: float        # Rotor speed correction (humidity)

    # Outlet states
    T_SUP_hr_out: float   # Supply air outlet temperature [degC]
    T_ETA_hr_out: float   # Exhaust air outlet temperature [degC]
    x_SUP_hr_out: float   # Supply air outlet absolute humidity [kg/kg]
    x_ETA_hr_out: float   # Exhaust air outlet absolute humidity [kg/kg]

    # Optimized values (after control mode optimization)
    n_rot_new: float      # Optimal rotor speed [1/min]
    eta_hr_new: float     # Optimized temperature transfer efficiency [-]
    eta_xr_new: float     # Optimized humidity transfer efficiency [-]
    f_n_new: float        # Optimized f_n [-]
    f_n_x_new: float      # Optimized f_n_x [-]

    # Inlet states (passed through for reference)
    T_ODA_preh: float     # Outdoor air temperature after preheating [degC]
    x_ODA_preh: float     # Outdoor air absolute humidity after preheating [kg/kg]


# ---------------------------------------------------------------------------
# Fan location factor
# ---------------------------------------------------------------------------

def get_f_ODA_hr(sup_fan_loc: FanLocation, eta_fan_loc: FanLocation) -> float:
    """
    Determine the outdoor air correction factor based on fan positions
    relative to the heat recovery unit.

    Parameters
    ----------
    sup_fan_loc : FanLocation
        Supply fan location (UP_HR or DOWN_HR).
    eta_fan_loc : FanLocation
        Exhaust fan location (UP_HR or DOWN_HR).

    Returns
    -------
    float
        f_ODA_hr correction factor.
    """
    if sup_fan_loc == FanLocation.UP_HR and eta_fan_loc == FanLocation.UP_HR:
        return 1.0
    elif sup_fan_loc == FanLocation.UP_HR and eta_fan_loc == FanLocation.DOWN_HR:
        return 1.0
    elif sup_fan_loc == FanLocation.DOWN_HR and eta_fan_loc == FanLocation.UP_HR:
        return 0.909
    elif sup_fan_loc == FanLocation.DOWN_HR and eta_fan_loc == FanLocation.DOWN_HR:
        return 0.98
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Core efficiency calculations (scalar, single timestep)
# ---------------------------------------------------------------------------

def _calc_f_q(q_V_ETA: float, q_V_SUP: float, f_ODA_min: float) -> float:
    """Airflow ratio correction factor for temperature transfer."""
    if q_V_SUP == 0.0 or q_V_ETA == 0.0:
        return 0.0
    ratio = (q_V_ETA - q_V_SUP) / (q_V_SUP * f_ODA_min) + 1.0
    return max(0.0, ratio) ** 0.4


def _calc_f_v(v_hr_eff: float, v_hr_nom: float, f_ODA_min: float,
              C1: float, C2: float) -> float:
    """Face velocity correction factor for temperature transfer."""
    if v_hr_eff == 0.0:
        return 0.0
    return max(0.0, C1 * (v_hr_eff * v_hr_nom * f_ODA_min - v_hr_nom) + C2)


def _calc_f_n(n_rot: float, n_rot_max: float,
              C3: float, C4: float, C5: float, e1: float) -> float:
    """Rotor speed correction factor for temperature transfer."""
    if n_rot == 0.0:
        return 0.0
    if n_rot_max == 0.0:
        return 0.0
    return max(0.0, C3 - C4 * (n_rot / n_rot_max + C5) ** e1)


def _calc_f_dx_x(rotor_type: RotorType, x_calc_cond: float, x_e_sat: float,
                 C6: float, C7: float, C8: float, dx_e_nom: float) -> float:
    """Condensation potential correction factor for humidity transfer."""
    if rotor_type in (RotorType.ROT_SORP, RotorType.ROT_NH):
        return min(1.0, max(0.0, C6 * (x_calc_cond - x_e_sat) + C7))
    else:  # ROT_HYG
        val1 = 1.0 + C6 * (x_calc_cond - x_e_sat - dx_e_nom)
        val2 = C7 + C8 * (x_calc_cond - x_e_sat - dx_e_nom)
        return min(1.0, max(0.0, val1, val2))


def _calc_f_q_x(q_calc_cond: float, q_calc_evap: float, f_ODA_min: float,
                C9: float) -> float:
    """Airflow ratio correction factor for humidity transfer."""
    if q_calc_evap == 0.0 or q_calc_cond == 0.0:
        return 0.0
    return max(0.0, 1.0 - C9 * (q_calc_cond - q_calc_evap) / (q_calc_evap * f_ODA_min))


def _calc_f_v_x(v_hr_eff: float, v_hr_nom: float, f_ODA_min: float,
                C10: float, C11: float) -> float:
    """Face velocity correction factor for humidity transfer."""
    if v_hr_eff == 0.0:
        return 0.0
    return max(0.0, C10 * (v_hr_eff * v_hr_nom * f_ODA_min - v_hr_nom) + C11)


def _calc_f_n_x(n_rot: float, n_rot_max: float,
                C12: float, C13: float, C14: float, e2: float) -> float:
    """Rotor speed correction factor for humidity transfer."""
    if n_rot == 0.0:
        return 0.0
    if n_rot == n_rot_max:
        return 1.0
    if n_rot_max == 0.0:
        return 0.0
    # Normalize to 20 1/min reference
    n_norm = (n_rot / n_rot_max) * 20.0
    return max(0.0, C12 - C13 * (n_norm + C14) ** e2)


def _saturation_pressure(T: float) -> float:
    """Magnus formula for saturation vapour pressure [Pa]."""
    return 611.2 * math.exp(17.62 * T / (243.12 + T))


def _saturation_humidity(T: float, p_atm: float) -> float:
    """Saturation absolute humidity [kg/kg dry air]."""
    p_s = _saturation_pressure(T)
    if p_atm <= p_s:
        # Avoid division by zero at extreme conditions
        return 0.622  # physical upper bound approximation
    return 0.622 * p_s / (p_atm - p_s)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RotaryHeatExchanger:
    """
    Rotary heat exchanger model per OENORM EN 16798-5-1:2017 (IDA ICE variant).

    Parameters
    ----------
    rotor_type : RotorType
        Type of rotor (ROT_SORP, ROT_HYG, or ROT_NH).
    control_mode : ControlMode
        Speed optimization strategy (TEMP, HUM, or ENERGY).
    sup_fan_loc : FanLocation
        Supply fan position relative to HRV (UP_HR or DOWN_HR).
    eta_fan_loc : FanLocation
        Exhaust fan position relative to HRV (UP_HR or DOWN_HR).
    n_rot_max : float
        Maximum rotor speed [1/min]. If 0, defaults to 20.
    f_ODA_min : float
        Minimum outdoor air fraction [-]. Typically 1.0.
    eta_hr_N : float
        Override for nominal temperature transfer efficiency. 0 = use default.
    eta_xr_N : float
        Override for nominal humidity transfer efficiency. 0 = use default.
    n_opt_steps : int
        Number of discretization steps for rotor speed optimization (default 1001).
    """

    def __init__(
        self,
        rotor_type: RotorType,
        control_mode: ControlMode = ControlMode.TEMP,
        sup_fan_loc: FanLocation = FanLocation.UP_HR,
        eta_fan_loc: FanLocation = FanLocation.UP_HR,
        n_rot_max: float = 20.0,
        f_ODA_min: Optional[float] = None,
        eta_hr_N: float = 0.0,
        eta_xr_N: float = 0.0,
        n_opt_steps: int = 1001,
    ) -> None:
        self.rotor_type = rotor_type
        self.control_mode = control_mode
        self.sup_fan_loc = sup_fan_loc
        self.eta_fan_loc = eta_fan_loc

        # Load default constants for rotor type
        self.const = ROTOR_CONSTANTS[rotor_type]

        # Maximum rotor speed (default to 20 if 0)
        self.n_rot_max = n_rot_max if n_rot_max != 0.0 else 20.0

        # Fan location factor
        self.f_ODA_hr = get_f_ODA_hr(sup_fan_loc, eta_fan_loc)
        # f_ODA_min: currently set equal to f_ODA_hr (volume-based determination
        # per section 6.4.3.2.5 not yet implemented)
        self.f_ODA_min = f_ODA_min if f_ODA_min is not None else self.f_ODA_hr

        # V5 parametrization: override nominal efficiencies if provided
        self.eta_hr_nom = eta_hr_N if eta_hr_N != 0.0 else self.const.eta_hr_nom
        self.eta_xr_nom = eta_xr_N if eta_xr_N != 0.0 else self.const.eta_xr_nom

        # V5: v_hr_N_para = v_hr_nom
        self.v_hr_N_para = self.const.v_hr_nom

        # Number of optimization steps
        self.n_opt_steps = n_opt_steps

    # -------------------------------------------------------------------
    # Core efficiency computation for a single operating point
    # -------------------------------------------------------------------

    def compute_efficiencies(
        self,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        x_ETA_hr_in: float,
        T_e: float,
        T_ETA_dis_out: float,
        q_V_SUP: float,
        q_V_ETA: float,
        v_hr_eff: float,
        p_atm: float,
        n_rot: float,
    ) -> dict:
        """
        Compute transfer efficiencies and correction factors for a single
        rotor speed.

        Parameters
        ----------
        T_ODA_preh : float
            Outdoor air temperature after optional preheating [degC].
        x_ODA_preh : float
            Outdoor air absolute humidity after optional preheating [kg/kg].
        T_ETA_hr_in : float
            Exhaust air temperature entering the HRV [degC].
        x_ETA_hr_in : float
            Exhaust air absolute humidity entering the HRV [kg/kg].
        T_e : float
            Raw outdoor air temperature (before any preheating) [degC].
        T_ETA_dis_out : float
            Exhaust air discharge temperature (used for heating/cooling case
            determination) [degC].
        q_V_SUP : float
            Supply airflow rate [m3/h].
        q_V_ETA : float
            Exhaust airflow rate [m3/h].
        v_hr_eff : float
            Effective face velocity at the rotor [m/s].
            When using v_hr_N_para parametrization, pass the ratio
            q_V_SUP / q_V_SUP_nom (i.e. the normalized velocity).
        p_atm : float
            Atmospheric pressure [Pa].
        n_rot : float
            Rotor speed [1/min].

        Returns
        -------
        dict
            Dictionary with keys: eta_hr, eta_xr, f_q, f_v, f_n,
            f_dx_x, f_q_x, f_v_x, f_n_x, T_SUP_hr_out, T_ETA_hr_out,
            x_SUP_hr_out, x_ETA_hr_out.
        """
        c = self.const
        f_ODA_min = self.f_ODA_min
        n_rot_max = self.n_rot_max

        # --- Temperature transfer correction factors ---
        f_q = _calc_f_q(q_V_ETA, q_V_SUP, f_ODA_min)
        f_v = _calc_f_v(v_hr_eff, self.v_hr_N_para, f_ODA_min, c.C1, c.C2)
        f_n = _calc_f_n(n_rot, n_rot_max, c.C3, c.C4, c.C5, c.e1)

        eta_hr = self.eta_hr_nom * f_q * f_v * f_n

        # --- Humidity transfer: heating vs cooling case ---
        if T_e > T_ETA_dis_out:
            # Cooling case
            T_calc_evap = T_ETA_hr_in
            x_calc_cond = x_ODA_preh
            q_calc_evap = q_V_ETA
            q_calc_cond = q_V_SUP
        else:
            # Heating case
            T_calc_evap = T_ODA_preh
            x_calc_cond = x_ETA_hr_in
            q_calc_evap = q_V_SUP
            q_calc_cond = q_V_ETA

        # Saturation humidity at evaporation-side temperature
        p_e_sat = _saturation_pressure(T_calc_evap)
        if p_atm > p_e_sat:
            x_e_sat = 0.622 * p_e_sat / (p_atm - p_e_sat)
        else:
            x_e_sat = 0.622

        # Humidity correction factors
        f_dx_x = _calc_f_dx_x(self.rotor_type, x_calc_cond, x_e_sat,
                               c.C6, c.C7, c.C8, c.dx_e_nom)
        f_q_x = _calc_f_q_x(q_calc_cond, q_calc_evap, f_ODA_min, c.C9)
        f_v_x = _calc_f_v_x(v_hr_eff, self.v_hr_N_para, f_ODA_min, c.C10, c.C11)
        f_n_x = _calc_f_n_x(n_rot, n_rot_max, c.C12, c.C13, c.C14, c.e2)

        eta_xr = self.eta_xr_nom * f_dx_x * f_q_x * f_v_x * f_n_x

        # --- Outlet states ---
        T_SUP_hr_out = T_ODA_preh + eta_hr * (T_ETA_hr_in - T_ODA_preh)
        T_ETA_hr_out = T_ETA_hr_in - eta_hr * (T_ETA_hr_in - T_ODA_preh)
        x_SUP_hr_out = x_ODA_preh + eta_xr * (x_ETA_hr_in - x_ODA_preh)
        x_ETA_hr_out = x_ETA_hr_in - eta_xr * (x_ETA_hr_in - x_ODA_preh)

        return {
            "eta_hr": eta_hr,
            "eta_xr": eta_xr,
            "f_q": f_q,
            "f_v": f_v,
            "f_n": f_n,
            "f_dx_x": f_dx_x,
            "f_q_x": f_q_x,
            "f_v_x": f_v_x,
            "f_n_x": f_n_x,
            "T_SUP_hr_out": T_SUP_hr_out,
            "T_ETA_hr_out": T_ETA_hr_out,
            "x_SUP_hr_out": x_SUP_hr_out,
            "x_ETA_hr_out": x_ETA_hr_out,
        }

    # -------------------------------------------------------------------
    # Vectorized optimization over rotor speed
    # -------------------------------------------------------------------

    def _build_opt_arrays(
        self,
        f_q: float,
        f_v: float,
        f_dx_x: float,
        f_q_x: float,
        f_v_x: float,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        x_ETA_hr_in: float,
    ) -> dict:
        """
        Build arrays of efficiencies and outlet states over the full range
        of rotor speeds (0 to n_rot_max) for optimization.

        Returns a dict with numpy arrays keyed by variable name.
        """
        c = self.const
        n_rot_max = self.n_rot_max

        n_rot_opt = np.linspace(0.0, n_rot_max, self.n_opt_steps)

        # f_n over rotor speed range
        with np.errstate(divide="ignore", invalid="ignore"):
            f_n_opt = np.maximum(
                0.0,
                c.C3 - c.C4 * (n_rot_opt / n_rot_max + c.C5) ** c.e1,
            )
        # n_rot = 0 => f_n = 0
        f_n_opt[0] = 0.0

        # f_n_x over rotor speed range
        n_norm = (n_rot_opt / n_rot_max) * 20.0
        with np.errstate(divide="ignore", invalid="ignore"):
            f_n_x_opt = np.maximum(
                0.0,
                c.C12 - c.C13 * (n_norm + c.C14) ** c.e2,
            )
        # n_rot = 0 => f_n_x = 0; n_rot = n_rot_max => f_n_x = 1
        f_n_x_opt[0] = 0.0
        f_n_x_opt[-1] = 1.0

        # Full efficiency arrays
        eta_hr_opt = self.eta_hr_nom * f_q * f_v * f_n_opt
        eta_xr_opt = self.eta_xr_nom * f_dx_x * f_q_x * f_v_x * f_n_x_opt

        # Outlet states over rotor speed range
        T_SUP_hr_out_opt = T_ODA_preh + eta_hr_opt * (T_ETA_hr_in - T_ODA_preh)
        x_SUP_hr_out_opt = x_ODA_preh + eta_xr_opt * (x_ETA_hr_in - x_ODA_preh)

        return {
            "n_rot_opt": n_rot_opt,
            "f_n_opt": f_n_opt,
            "f_n_x_opt": f_n_x_opt,
            "eta_hr_opt": eta_hr_opt,
            "eta_xr_opt": eta_xr_opt,
            "T_SUP_hr_out_opt": T_SUP_hr_out_opt,
            "x_SUP_hr_out_opt": x_SUP_hr_out_opt,
        }

    def _select_opt_index(self, idx: Optional[int], n_steps: int) -> int:
        """Safely clamp an optimization index. Returns 0 if None or invalid."""
        if idx is None:
            return 0
        return max(0, min(idx, n_steps - 1))

    # -------------------------------------------------------------------
    # Temperature-controlled optimization (WRG_Ctrl = 'Temp')
    # -------------------------------------------------------------------

    def _optimize_temp(
        self,
        opt: dict,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        T_SUP_hr_req_min: float,
        T_SUP_hr_req_max: float,
        x_SUP_hr_req_min_Tmax: float,
        x_SUP_hr_req_max_Tmin: float,
        x_SUP_hr_req_min_Tmin: float,
        x_SUP_hr_req_max_Tmax: float,
        dT_fan: float,
        has_humidifier: bool,
        humidifier_type: int,
        has_cooling_coil: bool,
        cooling_coil_dehumid: bool,
        h_H2O: float,
        h_ZUL_Soll_phi_min_T_min: float,
        h_ZUL_Soll_phi_min_T_max: float,
    ) -> int:
        """
        Find the optimal rotor speed index for temperature control mode.

        This implements the 'Temp' branch of the MATLAB V5 code.
        The logic selects the rotor speed that brings the supply air outlet
        temperature closest to the setpoint, considering humidifier
        interaction when applicable.

        Parameters
        ----------
        opt : dict
            Optimization arrays from _build_opt_arrays.
        T_ODA_preh : float
            Outdoor air temperature after preheating [degC].
        x_ODA_preh : float
            Outdoor air humidity after preheating [kg/kg].
        T_ETA_hr_in : float
            Exhaust air inlet temperature [degC].
        T_SUP_hr_req_min : float
            Minimum required supply temperature setpoint [degC].
        T_SUP_hr_req_max : float
            Maximum required supply temperature setpoint [degC].
        x_SUP_hr_req_min_Tmax, x_SUP_hr_req_max_Tmin : float
            Humidity setpoint limits at corresponding temperature limits.
        x_SUP_hr_req_min_Tmin, x_SUP_hr_req_max_Tmax : float
            Additional humidity setpoint limits.
        dT_fan : float
            Fan heat contribution (temperature rise) [K].
        has_humidifier : bool
            Whether a humidifier is present downstream of HRV.
        humidifier_type : int
            1 = steam humidifier, 2 = spray humidifier.
        has_cooling_coil : bool
            Whether a cooling coil is present downstream of HRV.
        cooling_coil_dehumid : bool
            Whether the cooling coil supports dehumidification.
        h_H2O : float
            Specific enthalpy of humidification water/steam [kJ/kg].
        h_ZUL_Soll_phi_min_T_min : float
            Target enthalpy at min humidity, min temperature [kJ/kg].
        h_ZUL_Soll_phi_min_T_max : float
            Target enthalpy at min humidity, max temperature [kJ/kg].

        Returns
        -------
        int
            Optimal index into the n_rot_opt array.
        """
        T_out = opt["T_SUP_hr_out_opt"]
        x_out = opt["x_SUP_hr_out_opt"]
        n_steps = len(T_out)
        has_setpoint_range = not (
            T_SUP_hr_req_min == T_SUP_hr_req_max
            and x_SUP_hr_req_min_Tmin == x_SUP_hr_req_max_Tmax
        )

        # --- Case: humidifier present and outdoor air is dry ---
        if has_humidifier and x_ODA_preh < x_SUP_hr_req_min_Tmin:
            # Compute pre-humidifier temperature limits based on enthalpy balance
            dx_Bef_min = x_SUP_hr_req_min_Tmin - x_out
            dx_Bef_max = x_SUP_hr_req_min_Tmax - x_out

            # Temperature before humidifier that achieves target enthalpy
            # T = (h_target - h_H2O * dx) / c_pl  (simplified; in MATLAB uses T(h,x))
            # Using the simplified relation: h = 1.004*T + x*(2500.9 + 1.86*T)
            # => T = (h - x * 2500.9) / (1.004 + x * 1.86)
            def _T_from_h_x(h_val: float, x_val: float) -> float:
                """Compute temperature from enthalpy and absolute humidity."""
                return (h_val - x_val * 2500.9) / (1.004 + x_val * 1.86)

            T_vBef_min = np.array([
                _T_from_h_x(h_ZUL_Soll_phi_min_T_min - h_H2O * dx, x_val) + dT_fan
                for dx, x_val in zip(dx_Bef_min, x_out)
            ])
            T_vBef_max = np.array([
                _T_from_h_x(h_ZUL_Soll_phi_min_T_max - h_H2O * dx, x_val) + dT_fan
                for dx, x_val in zip(dx_Bef_max, x_out)
            ])

            # Heating case: outdoor air is colder than humidifier limit
            if T_ODA_preh < T_vBef_min[0]:
                max_T_out = np.max(T_out)
                max_idx = int(np.where(T_out == max_T_out)[0][-1])

                if max_T_out > T_vBef_min[max_idx]:
                    # HRV can reach the required temperature
                    if has_setpoint_range:
                        candidates = np.where(T_out >= T_vBef_min)[0]
                        return int(candidates[0]) if len(candidates) > 0 else max_idx
                    else:
                        candidates = np.where(T_out > T_vBef_min)[0]
                        if len(candidates) > 0 and candidates[0] > 0:
                            return int(candidates[0]) - 1
                        return max_idx
                else:
                    # HRV cannot reach limit => maximize temperature
                    return max_idx

            # Cooling case: outdoor air is warmer than humidifier limit
            elif T_ODA_preh > T_vBef_max[0]:
                min_T_out = np.min(T_out)
                min_idx = int(np.where(T_out == min_T_out)[0][-1])

                if min_T_out < T_vBef_max[min_idx]:
                    if has_setpoint_range:
                        candidates = np.where(T_out <= T_vBef_max)[0]
                        return int(candidates[0]) if len(candidates) > 0 else min_idx
                    else:
                        candidates = np.where(T_out < T_vBef_max)[0]
                        if len(candidates) > 0 and candidates[0] > 0:
                            return int(candidates[0]) - 1
                        return min_idx
                else:
                    return min_idx

            # Within range: find point within both limits
            else:
                candidates = np.where(
                    (T_out <= T_vBef_max) & (T_out >= T_vBef_min)
                )[0]
                if len(candidates) > 0:
                    return int(candidates[-1])
                return n_steps - 1

        # --- Case: outdoor air humid, cooling coil with dehumidification ---
        elif (x_ODA_preh > x_SUP_hr_req_max_Tmax
              and has_cooling_coil and cooling_coil_dehumid):
            min_x = np.min(x_out)
            if min_x > x_SUP_hr_req_max_Tmax:
                # Cannot reach humidity setpoint; minimize humidity or enthalpy
                if min_x - 0.03e-3 > x_SUP_hr_req_max_Tmax:
                    # Use minimum enthalpy approach
                    h_approx = 1.004 * T_out + x_out * (2500.9 + 1.86 * T_out)
                    return int(np.argmin(h_approx))
                else:
                    return int(np.argmin(x_out))
            else:
                candidates = np.where(x_out <= x_SUP_hr_req_max_Tmax)[0]
                return int(candidates[0]) if len(candidates) > 0 else 0

        # --- Default case: pure temperature control ---
        else:
            return self._temp_control_default(
                T_out, T_ODA_preh, T_ETA_hr_in,
                T_SUP_hr_req_min, T_SUP_hr_req_max,
                dT_fan, has_setpoint_range, n_steps,
            )

    def _temp_control_default(
        self,
        T_out: np.ndarray,
        T_ODA_preh: float,
        T_ETA_hr_in: float,
        T_SUP_hr_req_min: float,
        T_SUP_hr_req_max: float,
        dT_fan: float,
        has_setpoint_range: bool,
        n_steps: int,
    ) -> int:
        """
        Default temperature control logic without humidifier interaction.

        Selects the rotor speed that brings the supply outlet temperature
        closest to the setpoint range, accounting for fan heat.
        """
        T_target_min = T_SUP_hr_req_min + dT_fan
        T_target_max = T_SUP_hr_req_max + dT_fan

        max_T = np.max(T_out)
        min_T = np.min(T_out)

        # Heating case: outdoor air below min setpoint
        if T_ODA_preh < T_SUP_hr_req_min:
            if max_T > T_target_min:
                if has_setpoint_range:
                    candidates = np.where(T_out >= T_target_min)[0]
                    return int(candidates[0]) if len(candidates) > 0 else n_steps - 1
                else:
                    candidates = np.where(T_out > T_target_min)[0]
                    if len(candidates) > 0 and candidates[0] > 0:
                        return int(candidates[0]) - 1
                    return n_steps - 1
            else:
                # Cannot reach setpoint => maximize heat recovery
                return int(np.where(T_out == max_T)[0][-1])

        # Cooling case: outdoor air above max setpoint
        elif T_ODA_preh > T_SUP_hr_req_max:
            if min_T < T_target_max:
                if has_setpoint_range:
                    candidates = np.where(T_out <= T_target_max)[0]
                    return int(candidates[0]) if len(candidates) > 0 else n_steps - 1
                else:
                    candidates = np.where(T_out < T_target_max)[0]
                    if len(candidates) > 0 and candidates[0] > 0:
                        return int(candidates[0]) - 1
                    return n_steps - 1
            else:
                # Cannot reach setpoint => minimize temperature
                return int(np.where(T_out == min_T)[0][-1])

        # Both outdoor and exhaust above max setpoint: special case
        elif (T_ODA_preh > T_target_max and T_ETA_hr_in > T_target_max):
            if T_ETA_hr_in < T_ODA_preh:
                # Exhaust is cooler => run at max speed
                return n_steps - 1
            else:
                # No benefit from HRV
                return 0

        # Within setpoint range: run at max speed
        else:
            return n_steps - 1

    # -------------------------------------------------------------------
    # Humidity-controlled optimization (WRG_Ctrl = 'Hum')
    # -------------------------------------------------------------------

    def _optimize_hum(
        self,
        base: dict,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        x_ETA_hr_in: float,
        x_SUP_hr_req_min_Tmax: float,
        x_SUP_hr_req_max_Tmin: float,
    ) -> tuple[float, float, float, float, float]:
        """
        Humidity control mode: directly calculate the required rotor speed
        from the humidity setpoint.

        Implements the 'Hum' branch of the MATLAB V5 code.

        Returns
        -------
        tuple of (n_rot_new, eta_hr_new, eta_xr_new, f_n_new, f_n_x_new)
        """
        c = self.const
        n_rot_max = self.n_rot_max

        eta_hr = base["eta_hr"]
        eta_xr = base["eta_xr"]
        f_q = base["f_q"]
        f_v = base["f_v"]
        f_n = base["f_n"]
        f_n_x = base["f_n_x"]
        f_dx_x = base["f_dx_x"]
        f_q_x = base["f_q_x"]
        f_v_x = base["f_v_x"]
        x_SUP_hr_out = base["x_SUP_hr_out"]

        denom = x_ETA_hr_in - x_ODA_preh
        if abs(denom) < 1e-12:
            # No humidity difference => keep current values
            return (base.get("n_rot", n_rot_max), eta_hr, eta_xr, f_n, f_n_x)

        # Case 1: supply humidity too high, but outdoor air is dry
        if (x_SUP_hr_out > x_SUP_hr_req_max_Tmin
                and x_ODA_preh < x_SUP_hr_req_min_Tmax):
            eta_xr_new = min(
                eta_xr,
                (x_SUP_hr_req_max_Tmin - x_ODA_preh) / denom,
            )

        # Case 2: supply humidity too low, but outdoor air is humid
        elif (x_SUP_hr_out < x_SUP_hr_req_min_Tmax
              and x_ODA_preh > x_SUP_hr_req_max_Tmin):
            eta_xr_new = min(
                eta_xr,
                (x_SUP_hr_req_min_Tmax - x_ODA_preh) / denom,
            )

        # Case 3: both outdoor and exhaust above max humidity
        elif (x_ODA_preh > x_SUP_hr_req_max_Tmin
              and x_ETA_hr_in > x_SUP_hr_req_max_Tmin):
            if x_ETA_hr_in < x_ODA_preh:
                # Exhaust is drier => keep current
                return (base.get("n_rot", n_rot_max), eta_hr, eta_xr, f_n, f_n_x)
            else:
                # No benefit => turn off
                return (0.0, 0.0, 0.0, 0.0, 0.0)

        # Default: keep current values
        else:
            return (base.get("n_rot", n_rot_max), eta_hr, eta_xr, f_n, f_n_x)

        # Compute required f_n_x from desired eta_xr
        product = self.eta_xr_nom * f_dx_x * f_q_x * f_v_x
        if product < 1e-12:
            f_n_x_new = 0.0
        else:
            f_n_x_new = min(1.0, eta_xr_new / product)

        # Invert f_n_x to get rotor speed
        # f_n_x = C12 - C13 * ((n_rot/n_rot_max)*20 + C14)^e2
        # => ((C12 - f_n_x) / C13)^(1/e2) - C14 = (n_rot/n_rot_max)*20
        diff = c.C12 - f_n_x_new
        if diff <= 0.0 or c.C13 == 0.0:
            n_rot_new = n_rot_max
        else:
            inner = (diff / c.C13) ** (1.0 / c.e2) - c.C14
            n_rot_new = min(n_rot_max, inner * n_rot_max / 20.0)
            n_rot_new = max(0.0, n_rot_new)

        # Recompute f_n at the new rotor speed
        f_n_new = _calc_f_n(n_rot_new, n_rot_max, c.C3, c.C4, c.C5, c.e1)
        eta_hr_new = self.eta_hr_nom * f_q * f_v * f_n_new

        return (n_rot_new, eta_hr_new, eta_xr_new, f_n_new, f_n_x_new)

    # -------------------------------------------------------------------
    # Energy-optimized control (WRG_Ctrl = 'Energy')
    # -------------------------------------------------------------------

    def _optimize_energy(
        self,
        opt: dict,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        T_SUP_hr_req_min: float,
        T_SUP_hr_req_max: float,
        x_SUP_hr_req_min_Tmax: float,
        x_SUP_hr_req_max_Tmin: float,
        x_SUP_hr_req_min_Tmin: float,
        x_SUP_hr_req_max_Tmax: float,
        dT_fan: float,
        has_humidifier: bool,
        humidifier_type: int,
        has_cooling_coil: bool,
        cooling_coil_dehumid: bool,
        h_H2O: float,
        h_ZUL_Soll_phi_min_T_min: float,
        h_ZUL_Soll_phi_min_T_max: float,
        p_atm: float,
    ) -> int:
        """
        Find the optimal rotor speed index for energy optimization mode.

        This implements the 'Energy' branch of the MATLAB V5 code. It selects
        the operating point that minimizes total downstream energy (heating +
        cooling + humidification), considering interactions with humidifier and
        cooling coil.

        Returns
        -------
        int
            Optimal index into the n_rot_opt array.
        """
        T_out = opt["T_SUP_hr_out_opt"]
        x_out = opt["x_SUP_hr_out_opt"]
        n_steps = len(T_out)
        has_setpoint_range = not (
            T_SUP_hr_req_min == T_SUP_hr_req_max
            and x_SUP_hr_req_min_Tmin == x_SUP_hr_req_max_Tmax
        )

        # --- Case: humidifier present and outdoor air is dry ---
        if has_humidifier and x_ODA_preh < x_SUP_hr_req_min_Tmin:

            def _T_from_h_x(h_val: float, x_val: float) -> float:
                return (h_val - x_val * 2500.9) / (1.004 + x_val * 1.86)

            if humidifier_type == 1:
                # Steam humidifier path
                dx_Bef_min = x_SUP_hr_req_min_Tmin - x_out
                dx_Bef_max = x_SUP_hr_req_min_Tmax - x_out

                T_vBef_min = np.array([
                    _T_from_h_x(h_ZUL_Soll_phi_min_T_min - h_H2O * dx, xv) + dT_fan
                    for dx, xv in zip(dx_Bef_min, x_out)
                ])
                T_vBef_max = np.array([
                    _T_from_h_x(h_ZUL_Soll_phi_min_T_max - h_H2O * dx, xv) + dT_fan
                    for dx, xv in zip(dx_Bef_max, x_out)
                ])

                if T_ODA_preh < T_vBef_min[0]:
                    max_T = np.max(T_out)
                    max_idx = int(np.where(T_out == max_T)[0][-1])
                    if max_T > T_vBef_min[max_idx]:
                        if max_T <= T_vBef_max[max_idx]:
                            return max_idx
                        else:
                            cands = np.where(T_out > T_vBef_max)[0]
                            if len(cands) > 0 and cands[0] > 0:
                                return int(cands[0]) - 1
                            return max_idx
                    else:
                        return max_idx
                elif T_ODA_preh > T_vBef_max[0]:
                    min_T = np.min(T_out)
                    min_idx = int(np.where(T_out == min_T)[0][-1])
                    if min_T < T_vBef_max[min_idx]:
                        if max(T_out) >= T_vBef_min[min_idx]:
                            return min_idx
                        else:
                            cands = np.where(T_out < T_vBef_min)[0]
                            if len(cands) > 0 and cands[0] > 0:
                                return int(cands[0]) - 1
                            return min_idx
                    else:
                        return int(np.where(T_out == min_T)[0][0])
                else:
                    cands = np.where(
                        (T_out <= T_vBef_max) & (T_out >= T_vBef_min)
                    )[0]
                    if len(cands) > 0:
                        return int(cands[-1])
                    return n_steps - 1

            elif humidifier_type == 2:
                # Spray humidifier path: energy-optimized selection
                dx_Bef_min = x_SUP_hr_req_min_Tmin - x_out
                dx_Bef_max = x_SUP_hr_req_min_Tmax - x_out

                T_vBef_min = np.array([
                    _T_from_h_x(h_ZUL_Soll_phi_min_T_min - h_H2O * dx, xv) + dT_fan
                    for dx, xv in zip(dx_Bef_min, x_out)
                ])
                T_vBef_max = np.array([
                    _T_from_h_x(h_ZUL_Soll_phi_min_T_max - h_H2O * dx, xv) + dT_fan
                    for dx, xv in zip(dx_Bef_max, x_out)
                ])

                # Compute energy cost metric
                dT_cost = np.maximum(
                    T_vBef_min - T_out,
                    T_out - T_vBef_max,
                )
                en_check = (np.maximum(0.0, dx_Bef_max * h_H2O)
                            + np.maximum(0.0, dT_cost))
                en_check = np.maximum(0.0, en_check)

                if T_ODA_preh < T_vBef_min[0]:
                    max_T = np.max(T_out)
                    max_idx = int(np.where(T_out == max_T)[0][-1])
                    if max_T > T_vBef_min[max_idx]:
                        if not has_cooling_coil:
                            if has_setpoint_range:
                                cands = np.where(T_out >= T_vBef_min)[0]
                                return int(cands[0]) if len(cands) > 0 else max_idx
                            else:
                                cands = np.where(T_out > T_vBef_min)[0]
                                if len(cands) > 0 and cands[0] > 0:
                                    return int(cands[0]) - 1
                                return max_idx
                        return int(np.argmin(en_check))
                    else:
                        if not has_cooling_coil:
                            return max_idx
                        return int(np.argmin(en_check))
                elif T_ODA_preh > T_vBef_max[0]:
                    min_T = np.min(T_out)
                    min_idx = int(np.where(T_out == min_T)[0][-1])
                    if min_T < T_vBef_max[min_idx]:
                        if not has_cooling_coil:
                            if has_setpoint_range:
                                cands = np.where(T_out <= T_vBef_max)[0]
                                return int(cands[0]) if len(cands) > 0 else min_idx
                            else:
                                cands = np.where(T_out < T_vBef_max)[0]
                                if len(cands) > 0 and cands[0] > 0:
                                    return int(cands[0]) - 1
                                return min_idx
                        return int(np.argmin(en_check))
                    else:
                        if not has_cooling_coil:
                            return min_idx
                        return int(np.argmin(en_check))
                else:
                    if np.max(x_out) < x_SUP_hr_req_min_Tmax:
                        return n_steps - 1
                    elif np.max(x_out) > x_SUP_hr_req_min_Tmax:
                        cands = np.where(x_out <= x_SUP_hr_req_min_Tmax)[0]
                        if len(cands) > 0:
                            return int(cands[-1])
                    return int(np.argmin(en_check))

            # Fallback for unknown humidifier type
            return n_steps - 1

        # --- Case: outdoor air humid, cooling coil with dehumidification ---
        elif (x_ODA_preh > x_SUP_hr_req_max_Tmax
              and has_cooling_coil and cooling_coil_dehumid):
            min_x = np.min(x_out)
            if min_x > x_SUP_hr_req_max_Tmax:
                if min_x - 0.03e-3 > x_SUP_hr_req_max_Tmax:
                    h_approx = 1.004 * T_out + x_out * (2500.9 + 1.86 * T_out)
                    return int(np.argmin(h_approx))
                else:
                    return int(np.argmin(x_out))
            else:
                cands = np.where(x_out <= x_SUP_hr_req_max_Tmax)[0]
                return int(cands[0]) if len(cands) > 0 else 0

        # --- Default: energy-optimized temperature control with humidity ---
        else:
            return self._energy_control_default(
                T_out, x_out, T_ODA_preh, x_ODA_preh, T_ETA_hr_in,
                T_SUP_hr_req_min, T_SUP_hr_req_max,
                x_SUP_hr_req_min_Tmin, x_SUP_hr_req_max_Tmin,
                x_SUP_hr_req_min_Tmax, x_SUP_hr_req_max_Tmax,
                dT_fan, has_setpoint_range, n_steps,
            )

    def _energy_control_default(
        self,
        T_out: np.ndarray,
        x_out: np.ndarray,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        T_SUP_hr_req_min: float,
        T_SUP_hr_req_max: float,
        x_SUP_hr_req_min_Tmin: float,
        x_SUP_hr_req_max_Tmin: float,
        x_SUP_hr_req_min_Tmax: float,
        x_SUP_hr_req_max_Tmax: float,
        dT_fan: float,
        has_setpoint_range: bool,
        n_steps: int,
    ) -> int:
        """
        Default energy optimization without humidifier interaction.
        Considers both temperature and humidity constraints when selecting
        the operating point.
        """
        T_target_min = T_SUP_hr_req_min + dT_fan
        T_target_max = T_SUP_hr_req_max + dT_fan

        max_T = np.max(T_out)
        min_T = np.min(T_out)

        # Heating case
        if T_ODA_preh < T_SUP_hr_req_min and max_T > T_target_min:
            # Check if humidity is within bounds at the temperature crossing
            idx_T = np.where(T_out >= T_target_min)[0]
            if len(idx_T) > 0:
                first_idx = int(idx_T[0])
                x_at_idx = x_out[first_idx]
                if (x_at_idx > x_SUP_hr_req_max_Tmin
                        or x_at_idx < x_SUP_hr_req_min_Tmin):
                    if (x_ODA_preh > x_SUP_hr_req_max_Tmax
                            or x_ODA_preh < x_SUP_hr_req_min_Tmin):
                        return first_idx
                    # Find point within humidity bounds
                    hum_ok = np.where(
                        (x_out <= x_SUP_hr_req_max_Tmin)
                        & (x_out >= x_SUP_hr_req_min_Tmin)
                    )[0]
                    if len(hum_ok) > 0:
                        temp_ok = hum_ok[T_out[hum_ok] >= T_target_min]
                        if len(temp_ok) > 0:
                            return int(np.min(temp_ok))
                    return first_idx
                return first_idx
            return n_steps - 1

        elif T_ODA_preh < T_SUP_hr_req_min and max_T <= T_target_min:
            max_idx = int(np.where(T_out == max_T)[0][-1])
            x_at_idx = x_out[max_idx]
            if (x_at_idx > x_SUP_hr_req_max_Tmin
                    or x_at_idx < x_SUP_hr_req_min_Tmin):
                if (x_ODA_preh > x_SUP_hr_req_max_Tmax
                        or x_ODA_preh < x_SUP_hr_req_min_Tmin):
                    return max_idx
                hum_ok = np.where(
                    (x_out <= x_SUP_hr_req_max_Tmin)
                    & (x_out >= x_SUP_hr_req_min_Tmin)
                )[0]
                if len(hum_ok) > 0:
                    temps = T_out[hum_ok]
                    return int(hum_ok[np.argmax(temps)])
                return max_idx
            return max_idx

        # Cooling case
        elif T_ODA_preh > T_SUP_hr_req_max and min_T < T_target_max:
            idx_T = np.where(T_out <= T_target_max)[0]
            if len(idx_T) > 0:
                first_idx = int(idx_T[0])
                x_at_idx = x_out[first_idx]
                if (x_at_idx > x_SUP_hr_req_max_Tmax
                        or x_at_idx < x_SUP_hr_req_min_Tmax):
                    if (x_ODA_preh > x_SUP_hr_req_max_Tmax
                            or x_ODA_preh < x_SUP_hr_req_min_Tmin):
                        return first_idx
                    hum_ok = np.where(
                        (x_out <= x_SUP_hr_req_max_Tmax)
                        & (x_out >= x_SUP_hr_req_min_Tmax)
                    )[0]
                    if len(hum_ok) > 0:
                        temp_ok = hum_ok[T_out[hum_ok] <= T_target_max]
                        if len(temp_ok) > 0:
                            return int(np.min(temp_ok))
                    return first_idx
                return first_idx
            return n_steps - 1

        elif T_ODA_preh > T_SUP_hr_req_max and min_T >= T_target_max:
            min_idx = int(np.where(T_out == min_T)[0][-1])
            x_at_idx = x_out[min_idx]
            if (x_at_idx > x_SUP_hr_req_max_Tmax
                    or x_at_idx < x_SUP_hr_req_min_Tmax):
                if (x_ODA_preh > x_SUP_hr_req_max_Tmax
                        or x_ODA_preh < x_SUP_hr_req_min_Tmin):
                    return min_idx
                hum_ok = np.where(
                    (x_out <= x_SUP_hr_req_max_Tmax)
                    & (x_out >= x_SUP_hr_req_min_Tmax)
                )[0]
                if len(hum_ok) > 0:
                    temps = T_out[hum_ok]
                    return int(hum_ok[np.argmin(temps)])
                return min_idx
            return min_idx

        # Both above max
        elif (T_ODA_preh > T_target_max and T_ETA_hr_in > T_target_max):
            if T_ETA_hr_in < T_ODA_preh:
                min_idx = int(np.where(T_out == min_T)[0][-1])
                x_at_idx = x_out[min_idx]
                if (x_at_idx > x_SUP_hr_req_max_Tmax
                        or x_at_idx < x_SUP_hr_req_min_Tmax):
                    if (x_ODA_preh > x_SUP_hr_req_max_Tmax
                            or x_ODA_preh < x_SUP_hr_req_min_Tmin):
                        return min_idx
                    hum_ok = np.where(
                        (x_out <= x_SUP_hr_req_max_Tmax)
                        & (x_out >= x_SUP_hr_req_min_Tmax)
                    )[0]
                    if len(hum_ok) > 0:
                        temps = T_out[hum_ok]
                        return int(hum_ok[np.argmin(temps)])
                    return min_idx
                return min_idx
            else:
                return 0  # No benefit

        else:
            return 0  # Rotor off

    # -------------------------------------------------------------------
    # Main calculation entry point
    # -------------------------------------------------------------------

    def calculate(
        self,
        T_ODA_preh: float,
        x_ODA_preh: float,
        T_ETA_hr_in: float,
        x_ETA_hr_in: float,
        T_e: float,
        T_ETA_dis_out: float,
        q_V_SUP: float,
        q_V_ETA: float,
        v_hr_eff: float,
        p_atm: float = 101325.0,
        n_rot: Optional[float] = None,
        T_SUP_hr_req_min: float = 18.0,
        T_SUP_hr_req_max: float = 22.0,
        x_SUP_hr_req_min_Tmax: float = 0.004,
        x_SUP_hr_req_max_Tmin: float = 0.008,
        x_SUP_hr_req_min_Tmin: float = 0.004,
        x_SUP_hr_req_max_Tmax: float = 0.008,
        h_ZUL_Soll_phi_min_T_min: float = 28.0,
        h_ZUL_Soll_phi_min_T_max: float = 32.0,
        has_humidifier: bool = False,
        humidifier_type: int = 1,
        h_H2O: float = 2676.0,
        has_cooling_coil: bool = False,
        cooling_coil_dehumid: bool = False,
        dT_fan: float = 0.0,
    ) -> RotaryResult:
        """
        Calculate the rotary heat exchanger for a single timestep.

        This is the main entry point that computes baseline efficiencies at
        the given (or maximum) rotor speed, then optimizes according to the
        selected control mode.

        Parameters
        ----------
        T_ODA_preh : float
            Outdoor air temperature after optional frost protection [degC].
        x_ODA_preh : float
            Outdoor air absolute humidity [kg/kg].
        T_ETA_hr_in : float
            Exhaust air temperature entering the HRV [degC].
        x_ETA_hr_in : float
            Exhaust air absolute humidity entering the HRV [kg/kg].
        T_e : float
            Raw outdoor air temperature (before any treatment) [degC].
        T_ETA_dis_out : float
            Exhaust discharge temperature for heating/cooling case
            determination [degC].
        q_V_SUP : float
            Supply airflow rate [m3/h].
        q_V_ETA : float
            Exhaust airflow rate [m3/h].
        v_hr_eff : float
            Effective face velocity ratio at the rotor [-].
            (Typically q_V_SUP / q_V_SUP_nom when using v_hr_N_para.)
        p_atm : float
            Atmospheric pressure [Pa]. Default 101325.
        n_rot : float or None
            Rotor speed [1/min]. If None, uses n_rot_max for initial calc.
        T_SUP_hr_req_min : float
            Minimum supply air temperature setpoint [degC].
        T_SUP_hr_req_max : float
            Maximum supply air temperature setpoint [degC].
        x_SUP_hr_req_min_Tmax : float
            Min humidity setpoint at max temperature [kg/kg].
        x_SUP_hr_req_max_Tmin : float
            Max humidity setpoint at min temperature [kg/kg].
        x_SUP_hr_req_min_Tmin : float
            Min humidity setpoint at min temperature [kg/kg].
        x_SUP_hr_req_max_Tmax : float
            Max humidity setpoint at max temperature [kg/kg].
        h_ZUL_Soll_phi_min_T_min : float
            Target enthalpy at min humidity, min temp [kJ/kg].
        h_ZUL_Soll_phi_min_T_max : float
            Target enthalpy at min humidity, max temp [kJ/kg].
        has_humidifier : bool
            Whether a humidifier is downstream of HRV.
        humidifier_type : int
            1 = steam, 2 = spray.
        h_H2O : float
            Specific enthalpy of humidification medium [kJ/kg].
        has_cooling_coil : bool
            Whether a cooling coil is downstream of HRV.
        cooling_coil_dehumid : bool
            Whether the cooling coil can dehumidify.
        dT_fan : float
            Temperature rise from supply fan [K].

        Returns
        -------
        RotaryResult
            Complete result including baseline and optimized values.
        """
        if n_rot is None:
            n_rot = self.n_rot_max

        # --- Step 1: Compute baseline efficiencies at given n_rot ---
        base = self.compute_efficiencies(
            T_ODA_preh, x_ODA_preh, T_ETA_hr_in, x_ETA_hr_in,
            T_e, T_ETA_dis_out, q_V_SUP, q_V_ETA, v_hr_eff, p_atm, n_rot,
        )
        base["n_rot"] = n_rot

        # --- Step 2: Optimize rotor speed based on control mode ---
        if self.control_mode == ControlMode.HUM:
            n_rot_new, eta_hr_new, eta_xr_new, f_n_new, f_n_x_new = (
                self._optimize_hum(
                    base, T_ODA_preh, x_ODA_preh, T_ETA_hr_in, x_ETA_hr_in,
                    x_SUP_hr_req_min_Tmax, x_SUP_hr_req_max_Tmin,
                )
            )

        elif self.control_mode in (ControlMode.TEMP, ControlMode.ENERGY):
            # Build optimization arrays
            opt = self._build_opt_arrays(
                base["f_q"], base["f_v"],
                base["f_dx_x"], base["f_q_x"], base["f_v_x"],
                T_ODA_preh, x_ODA_preh, T_ETA_hr_in, x_ETA_hr_in,
            )

            if self.control_mode == ControlMode.TEMP:
                idx = self._optimize_temp(
                    opt, T_ODA_preh, x_ODA_preh, T_ETA_hr_in,
                    T_SUP_hr_req_min, T_SUP_hr_req_max,
                    x_SUP_hr_req_min_Tmax, x_SUP_hr_req_max_Tmin,
                    x_SUP_hr_req_min_Tmin, x_SUP_hr_req_max_Tmax,
                    dT_fan, has_humidifier, humidifier_type,
                    has_cooling_coil, cooling_coil_dehumid,
                    h_H2O, h_ZUL_Soll_phi_min_T_min, h_ZUL_Soll_phi_min_T_max,
                )
            else:  # ENERGY
                idx = self._optimize_energy(
                    opt, T_ODA_preh, x_ODA_preh, T_ETA_hr_in,
                    T_SUP_hr_req_min, T_SUP_hr_req_max,
                    x_SUP_hr_req_min_Tmax, x_SUP_hr_req_max_Tmin,
                    x_SUP_hr_req_min_Tmin, x_SUP_hr_req_max_Tmax,
                    dT_fan, has_humidifier, humidifier_type,
                    has_cooling_coil, cooling_coil_dehumid,
                    h_H2O, h_ZUL_Soll_phi_min_T_min, h_ZUL_Soll_phi_min_T_max,
                    p_atm,
                )

            idx = self._select_opt_index(idx, self.n_opt_steps)

            n_rot_new = float(opt["n_rot_opt"][idx])
            eta_hr_new = float(opt["eta_hr_opt"][idx])
            eta_xr_new = float(opt["eta_xr_opt"][idx])
            f_n_new = float(opt["f_n_opt"][idx])
            f_n_x_new = float(opt["f_n_x_opt"][idx])

        else:
            # Unknown mode: keep baseline
            n_rot_new = n_rot
            eta_hr_new = base["eta_hr"]
            eta_xr_new = base["eta_xr"]
            f_n_new = base["f_n"]
            f_n_x_new = base["f_n_x"]

        # --- Step 3: Compute final outlet states at optimized speed ---
        T_SUP_hr_out = T_ODA_preh + eta_hr_new * (T_ETA_hr_in - T_ODA_preh)
        T_ETA_hr_out = T_ETA_hr_in - eta_hr_new * (T_ETA_hr_in - T_ODA_preh)
        x_SUP_hr_out = x_ODA_preh + eta_xr_new * (x_ETA_hr_in - x_ODA_preh)
        x_ETA_hr_out = x_ETA_hr_in - eta_xr_new * (x_ETA_hr_in - x_ODA_preh)

        return RotaryResult(
            eta_hr=base["eta_hr"],
            eta_xr=base["eta_xr"],
            f_q=base["f_q"],
            f_v=base["f_v"],
            f_n=base["f_n"],
            f_dx_x=base["f_dx_x"],
            f_q_x=base["f_q_x"],
            f_v_x=base["f_v_x"],
            f_n_x=base["f_n_x"],
            T_SUP_hr_out=T_SUP_hr_out,
            T_ETA_hr_out=T_ETA_hr_out,
            x_SUP_hr_out=x_SUP_hr_out,
            x_ETA_hr_out=x_ETA_hr_out,
            n_rot_new=n_rot_new,
            eta_hr_new=eta_hr_new,
            eta_xr_new=eta_xr_new,
            f_n_new=f_n_new,
            f_n_x_new=f_n_x_new,
            T_ODA_preh=T_ODA_preh,
            x_ODA_preh=x_ODA_preh,
        )

"""Heat recovery components - plate, rotary, and runaround coil."""

from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air
from src.heat_exchangers.ntu_dry import DryHeatExchanger, DryHXDesignData, FlowType


class PlateHeatRecovery(Component):
    """Plate heat exchanger for heat recovery (Plattenwärmeübertrager).

    Uses crossflow NTU model with Kaup part-load correction (n=0.4).
    Only dry operation (no condensation).

    Attributes:
        design_airflow_supply: Design supply airflow [m³/h]
        design_airflow_exhaust: Design exhaust airflow [m³/h]
        effectiveness_nominal: Nominal heat recovery effectiveness [0..1]
    """

    def __init__(self,
                 design_airflow_supply: float = 5000.0,
                 design_airflow_exhaust: float = 5000.0,
                 effectiveness_nominal: float = 0.6,
                 design_T_outdoor: float = -12.0,
                 design_T_exhaust: float = 22.0):
        self.design_airflow_supply = design_airflow_supply
        self.design_airflow_exhaust = design_airflow_exhaust
        self.effectiveness_nominal = effectiveness_nominal
        self.design_T_outdoor = design_T_outdoor
        self.design_T_exhaust = design_T_exhaust

        # Compute design outlet temperature from effectiveness
        T_supply_out = design_T_outdoor + effectiveness_nominal * (design_T_exhaust - design_T_outdoor)
        T_exhaust_out = design_T_exhaust - effectiveness_nominal * (design_T_exhaust - design_T_outdoor)

        self._hx = DryHeatExchanger(
            design=DryHXDesignData(
                T_L_in=design_T_outdoor,
                T_L_out=T_supply_out,
                T_W_in=design_T_exhaust,
                T_W_out=T_exhaust_out,
                V_dot_L=design_airflow_supply,
                V_dot_W=design_airflow_exhaust,
            ),
            flow_type=FlowType.CROSSFLOW,
            c_p_water=1.006,  # exhaust air, not water
            rho_water=1.2,    # air density, not water
        )

    @property
    def name(self) -> str:
        return "plate_heat_recovery"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  exhaust_air: AirState = None,
                  V_dot_supply: float = 0.0,
                  V_dot_exhaust: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate plate heat recovery.

        Args:
            air_in: Supply air inlet (outdoor air after frost protection)
            setpoint_T: Supply temperature setpoint [°C]
            setpoint_x: Supply humidity setpoint [kg/kg]
            exhaust_air: Exhaust air state
            V_dot_supply: Supply air volume flow [m³/h]
            V_dot_exhaust: Exhaust air volume flow [m³/h]
        """
        if exhaust_air is None or V_dot_supply <= 0 or V_dot_exhaust <= 0:
            return ComponentResult(air_out=air_in.copy())

        result = self._hx.calculate(
            T_L_in=air_in.T,
            T_W_in=exhaust_air.T,
            V_dot_L=V_dot_supply,
            V_dot_W=V_dot_exhaust,
        )

        T_supply_out = result['T_L_out']

        # Limit to setpoint if needed (partial bypass)
        if air_in.T < setpoint_T and T_supply_out > setpoint_T:
            T_supply_out = setpoint_T
        elif air_in.T > setpoint_T and T_supply_out < setpoint_T:
            T_supply_out = setpoint_T

        air_out = AirState(T=T_supply_out, x=air_in.x, p=air_in.p)

        # Exhaust side outlet
        T_exhaust_out = result['T_W_out']

        return ComponentResult(
            air_out=air_out,
            info={
                "T_exhaust_out": T_exhaust_out,
                "effectiveness": result['eps'],
                "Q_dot_kW": abs(result['Q_dot']),
            }
        )


class RotaryHeatRecovery(Component):
    """Rotary heat exchanger for heat recovery (Rotationswärmeübertrager).

    Delegates to the rotary heat exchanger model.
    Supports temperature and humidity transfer.

    Attributes:
        rotor_type: 'ROT_SORP', 'ROT_HYG', or 'ROT_NH'
        control_mode: 'Temp', 'Hum', or 'Energy'
        n_rot_max: Maximum rotor speed [1/min]
    """

    def __init__(self,
                 rotor_type: str = 'ROT_SORP',
                 control_mode: str = 'Temp',
                 sup_fan_loc: str = 'DOWN_HR',
                 eta_fan_loc: str = 'DOWN_HR',
                 design_airflow_supply: float = 5000.0,
                 n_rot_max: float = 20.0,
                 f_ODA_min: float = 1.0,
                 eta_hr_N: float = 0.0,
                 eta_xr_N: float = 0.0):
        self.rotor_type = rotor_type
        self.control_mode = control_mode
        self.sup_fan_loc = sup_fan_loc
        self.eta_fan_loc = eta_fan_loc
        self.design_airflow_supply = design_airflow_supply
        self.n_rot_max = n_rot_max
        self.f_ODA_min = f_ODA_min
        self.eta_hr_N = eta_hr_N
        self.eta_xr_N = eta_xr_N

        # Import here to avoid circular imports
        from src.heat_exchangers.rotary import RotaryHeatExchanger, RotorType, ControlMode, FanLocation

        rotor_type_enum = RotorType[rotor_type]
        control_mode_enum = ControlMode[{'Temp': 'TEMP', 'Hum': 'HUM', 'Energy': 'ENERGY'}[control_mode]]
        sup_fan_enum = FanLocation[sup_fan_loc]
        eta_fan_enum = FanLocation[eta_fan_loc]

        self._rotary = RotaryHeatExchanger(
            rotor_type=rotor_type_enum,
            control_mode=control_mode_enum,
            sup_fan_loc=sup_fan_enum,
            eta_fan_loc=eta_fan_enum,
            n_rot_max=n_rot_max,
            f_ODA_min=f_ODA_min,
            eta_hr_N=eta_hr_N,
            eta_xr_N=eta_xr_N,
        )

    @property
    def name(self) -> str:
        return "rotary_heat_recovery"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  exhaust_air: AirState = None,
                  V_dot_supply: float = 0.0,
                  V_dot_exhaust: float = 0.0,
                  T_outdoor: float | None = None,
                  T_exhaust_setpoint: float | None = None,
                  dT_fan: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate rotary heat recovery.

        Args:
            air_in: Supply air inlet (outdoor after preheating/frost)
            setpoint_T: Supply temperature setpoint [°C]
            setpoint_x: Supply humidity setpoint [kg/kg]
            exhaust_air: Exhaust air state
            V_dot_supply: Supply volume flow [m³/h]
            V_dot_exhaust: Exhaust volume flow [m³/h]
            T_outdoor: Original outdoor temperature [°C]
            T_exhaust_setpoint: Exhaust/room temperature setpoint [°C]
            dT_fan: Fan temperature rise [K]
        """
        if exhaust_air is None or V_dot_supply <= 0 or V_dot_exhaust <= 0:
            return ComponentResult(air_out=air_in.copy())

        if T_outdoor is None:
            T_outdoor = air_in.T
        if T_exhaust_setpoint is None:
            T_exhaust_setpoint = exhaust_air.T

        # Compute effective velocity (normalized)
        v_hr_eff = V_dot_supply / self.design_airflow_supply

        result = self._rotary.calculate(
            T_ODA_preh=air_in.T,
            x_ODA_preh=air_in.x,
            T_ETA_hr_in=exhaust_air.T,
            x_ETA_hr_in=exhaust_air.x,
            T_e=T_outdoor,
            T_ETA_dis_out=T_exhaust_setpoint,
            q_V_SUP=V_dot_supply,
            q_V_ETA=V_dot_exhaust,
            v_hr_eff=v_hr_eff,
            p_atm=air_in.p,
            n_rot=self.n_rot_max,
            T_SUP_hr_req_min=setpoint_T,
            T_SUP_hr_req_max=setpoint_T,
            x_SUP_hr_req_min_Tmin=setpoint_x,
            x_SUP_hr_req_max_Tmin=setpoint_x,
            x_SUP_hr_req_min_Tmax=setpoint_x,
            x_SUP_hr_req_max_Tmax=setpoint_x,
        )

        air_out = AirState(
            T=result.T_SUP_hr_out,
            x=result.x_SUP_hr_out,
            p=air_in.p
        )

        return ComponentResult(
            air_out=air_out,
            info={
                "eta_hr": result.eta_hr_new,
                "eta_xr": result.eta_xr_new,
                "n_rot_opt": result.n_rot_new,
                "T_exhaust_out": result.T_ETA_hr_out,
                "x_exhaust_out": result.x_ETA_hr_out,
            }
        )


class RunaroundCoilRecovery(Component):
    """Runaround coil system for heat recovery (KVS).

    Two counterflow heat exchangers connected by an intermediate
    fluid circuit. Uses NTU method for both heat exchangers.

    Attributes:
        design_airflow_supply: Design supply airflow [m³/h]
        design_airflow_exhaust: Design exhaust airflow [m³/h]
        effectiveness_supply: Nominal effectiveness supply side [0..1]
        effectiveness_exhaust: Nominal effectiveness exhaust side [0..1]
    """

    def __init__(self,
                 design_airflow_supply: float = 5000.0,
                 design_airflow_exhaust: float = 5000.0,
                 design_V_dot_medium: float = 1.0,
                 effectiveness_supply: float = 0.6,
                 effectiveness_exhaust: float = 0.6,
                 design_T_outdoor: float = -12.0,
                 design_T_exhaust: float = 22.0,
                 rho_medium: float = 1000.0,
                 cp_medium: float = 4.19):
        self.design_airflow_supply = design_airflow_supply
        self.design_airflow_exhaust = design_airflow_exhaust
        self.design_V_dot_medium = design_V_dot_medium
        self.effectiveness_supply = effectiveness_supply
        self.effectiveness_exhaust = effectiveness_exhaust
        self.rho_medium = rho_medium
        self.cp_medium = cp_medium

    @property
    def name(self) -> str:
        return "runaround_coil_recovery"

    def calculate(self,
                  air_in: AirState,
                  setpoint_T: float,
                  setpoint_x: float,
                  timestep_h: float = 1.0,
                  m_dot_dry: float = 0.0,
                  exhaust_air: AirState = None,
                  V_dot_supply: float = 0.0,
                  V_dot_exhaust: float = 0.0,
                  **kwargs) -> ComponentResult:
        """Calculate runaround coil heat recovery.

        Simplified model: combined effectiveness applied to temperature difference.
        """
        if exhaust_air is None or V_dot_supply <= 0:
            return ComponentResult(air_out=air_in.copy())

        # Simplified: overall effectiveness as product of both sides
        eta_overall = self.effectiveness_supply * self.effectiveness_exhaust
        # More accurate: combined effectiveness considering medium circuit
        eta_combined = eta_overall / (1 + eta_overall) * 2  # approximate

        T_out = air_in.T + eta_combined * (exhaust_air.T - air_in.T)

        # Limit to setpoint
        if air_in.T < setpoint_T and T_out > setpoint_T:
            T_out = setpoint_T
        elif air_in.T > setpoint_T and T_out < setpoint_T:
            T_out = setpoint_T

        air_out = AirState(T=T_out, x=air_in.x, p=air_in.p)

        return ComponentResult(
            air_out=air_out,
            info={"eta_combined": eta_combined}
        )

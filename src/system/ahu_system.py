"""AHU (Air Handling Unit) system orchestrator.

Chains all supply and exhaust air components together and computes
the aggregate energy demand for a single timestep.
"""

from dataclasses import dataclass, field
from src.components.base import Component
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air
from src.components.fan import Fan
from src.components.heat_recovery import PlateHeatRecovery, RotaryHeatRecovery, RunaroundCoilRecovery
from src.components.heating_coil import HeatingCoil
from src.components.cooling_coil import CoolingCoil
from src.components.humidifier import SprayHumidifier, SteamHumidifier

_HEAT_RECOVERY_TYPES = (PlateHeatRecovery, RotaryHeatRecovery, RunaroundCoilRecovery)
_FAN_CORRECTION_TYPES = (HeatingCoil, CoolingCoil)


@dataclass
class SystemResult:
    """Result of a single AHU timestep calculation.

    Attributes:
        air_supply_out: Final supply air state
        Q_heat: Total heating energy [kWh]
        Q_cool: Total cooling energy [kWh]
        W_el: Total electrical energy [kWh]
        m_water: Total water consumption [kg]
        component_results: Per-component breakdown {name: ComponentResult}
    """
    air_supply_out: AirState
    Q_heat: float = 0.0
    Q_cool: float = 0.0
    W_el: float = 0.0
    m_water: float = 0.0
    component_results: dict = field(default_factory=dict)


class AHUSystem:
    """Air Handling Unit system that processes air through a chain of components.

    Components are processed sequentially; the outlet of each component
    becomes the inlet of the next.

    Args:
        supply_air_components: Ordered list of supply-side components
        exhaust_air_components: Ordered list of exhaust-side components
        design_airflow_supply: Design supply airflow [m³/h]
        design_airflow_exhaust: Design exhaust airflow [m³/h]
        setpoint_T: Supply air temperature setpoint [°C]
        setpoint_x: Supply air absolute humidity setpoint [kg/kg]
        setpoint_T_range: Optional (min, max) temperature tolerance [°C]
        setpoint_x_range: Optional (min, max) humidity tolerance [kg/kg]
    """

    def __init__(self,
                 supply_air_components: list[Component],
                 exhaust_air_components: list[Component],
                 design_airflow_supply: float,
                 design_airflow_exhaust: float,
                 setpoint_T: float,
                 setpoint_x: float,
                 setpoint_T_range: tuple[float, float] | None = None,
                 setpoint_x_range: tuple[float, float] | None = None):
        self.supply_air_components = supply_air_components
        self.exhaust_air_components = exhaust_air_components
        self.design_airflow_supply = design_airflow_supply
        self.design_airflow_exhaust = design_airflow_exhaust
        self.setpoint_T = setpoint_T
        self.setpoint_x = setpoint_x
        self.setpoint_T_range = setpoint_T_range
        self.setpoint_x_range = setpoint_x_range

    def _find_downstream_humidifier(self, start_idx: int) -> SprayHumidifier | SteamHumidifier | None:
        """Find the next humidifier downstream of start_idx."""
        for comp in self.supply_air_components[start_idx + 1:]:
            if isinstance(comp, (SprayHumidifier, SteamHumidifier)):
                return comp
        return None

    def _compute_preheat_target_for_spray(
        self, air_in: AirState, setpoint_T: float, setpoint_x: float,
        spray: SprayHumidifier, dT_fan: float,
    ) -> float:
        """Compute VHR target temperature so spray humidifier reaches x_soll.

        For a spray humidifier, the pre-heating coil must heat the air high
        enough so that adiabatic humidification along the process line
        (h_out = h_in + dx * h_water) can reach setpoint_x before hitting
        the saturation curve.

        The key constraint: after humidification, the temperature should be
        at or below setpoint_T (since the NHR can only heat, not cool).

        Uses bisection to find T_preheat such that:
        spray_humidifier.calculate(AirState(T_preheat, x_in), ...).air_out.x ≈ setpoint_x
        """
        from scipy.optimize import brentq

        if air_in.x >= setpoint_x:
            return setpoint_T - dT_fan

        h_water = moist_air.water_enthalpy(spray.T_water)

        # Find the minimum T_preheat where spray can reach x_soll.
        # The residual is negative below the threshold and zero (flat) above it.
        # brentq can't distinguish between all the zero-residual points,
        # so we use a modified residual with a small positive offset above the
        # threshold, making it strictly monotonic.
        def spray_x_out(T_pre: float) -> float:
            """Return x_out from spray humidifier at given preheat temperature."""
            test_state = AirState(T=T_pre, x=air_in.x, p=air_in.p)
            res = spray.calculate(test_state, setpoint_T, setpoint_x, m_dot_dry=1.0)
            return res.air_out.x

        T_low = air_in.T
        T_high = 80.0

        # Check if even at T_high the spray can't reach setpoint_x
        if spray_x_out(T_high) < setpoint_x - 1e-6:
            return T_high

        # Check if no extra preheating is needed
        if spray_x_out(T_low) >= setpoint_x - 1e-6:
            return max(T_low, setpoint_T - dT_fan)

        # Binary search for the minimum T where x_out >= setpoint_x
        for _ in range(50):  # 50 iterations → precision < 0.01°C
            T_mid = (T_low + T_high) / 2.0
            if spray_x_out(T_mid) >= setpoint_x - 1e-6:
                T_high = T_mid
            else:
                T_low = T_mid
            if T_high - T_low < 0.01:
                break

        return T_high

    def _compute_preheat_target_for_steam(
        self, air_in: AirState, setpoint_T: float, setpoint_x: float,
        steam: SteamHumidifier, dT_fan: float,
    ) -> float:
        """Compute VHR target for steam humidifier.

        Steam humidification is near-isothermal, so we need to pre-heat
        so that after adding steam (slight T increase) we reach T_soll.
        Calculate: T_VHR = T such that after steam humidification from x_in
        to x_soll the resulting temperature equals setpoint_T - dT_fan.
        """
        if air_in.x >= setpoint_x:
            return setpoint_T - dT_fan

        dx = setpoint_x - air_in.x
        h_steam = moist_air.steam_enthalpy(steam.T_steam)
        # h_out = h_in + dx * h_steam, T_out = (h_out - 2500*x_out) / (1 + 1.85*x_out)
        # We want T_out = setpoint_T - dT_fan with x_out = setpoint_x
        T_target_after = setpoint_T - dT_fan
        h_target = moist_air.enthalpy(T_target_after, setpoint_x)
        h_needed_in = h_target - dx * h_steam
        T_pre = moist_air.temperature_from_h_x(h_needed_in, air_in.x)
        return T_pre

    def _find_downstream_fan(self, components: list[Component], start_idx: int) -> Fan | None:
        """Find the next fan downstream of start_idx in the component list."""
        for comp in components[start_idx + 1:]:
            if isinstance(comp, Fan):
                return comp
        return None

    def _compute_dT_fan(self, fan: Fan | None, m_dot_dry: float) -> float:
        """Compute the temperature rise from a downstream fan."""
        if fan is None:
            return 0.0
        return fan.temperature_rise(m_dot_dry)

    def _process_exhaust_components(self,
                                    exhaust_air: AirState,
                                    setpoint_T: float,
                                    setpoint_x: float,
                                    timestep_h: float,
                                    m_dot_dry_exhaust: float,
                                    V_dot_supply: float,
                                    V_dot_exhaust: float) -> tuple[AirState, dict[str, ComponentResult]]:
        """Process exhaust-side components (e.g. adiabatic cooling before HRV).

        Returns the exhaust air state entering the heat recovery unit and
        per-component results.
        """
        air = exhaust_air.copy()
        results = {}
        for comp in self.exhaust_air_components:
            if isinstance(comp, _HEAT_RECOVERY_TYPES):
                # Heat recovery on exhaust side is handled from supply side
                continue
            result = comp.calculate(
                air_in=air,
                setpoint_T=setpoint_T,
                setpoint_x=setpoint_x,
                timestep_h=timestep_h,
                m_dot_dry=m_dot_dry_exhaust,
            )
            air = result.air_out
            results[comp.name] = result
        return air, results

    def calculate_timestep(self,
                           outdoor_air: AirState,
                           exhaust_air: AirState,
                           V_dot_supply: float | None = None,
                           V_dot_exhaust: float | None = None,
                           timestep_h: float = 1.0) -> SystemResult:
        """Calculate one timestep through all components.

        Args:
            outdoor_air: Outdoor (fresh) air state
            exhaust_air: Exhaust (room) air state
            V_dot_supply: Supply airflow [m³/h], defaults to design
            V_dot_exhaust: Exhaust airflow [m³/h], defaults to design
            timestep_h: Timestep duration [h]

        Returns:
            SystemResult with totals and per-component breakdown
        """
        V_dot_supply = V_dot_supply or self.design_airflow_supply
        V_dot_exhaust = V_dot_exhaust or self.design_airflow_exhaust

        # Dry air mass flow rates
        m_dot_dry_supply = moist_air.mass_flow_from_volume_flow(
            V_dot_supply, outdoor_air.T, outdoor_air.x, outdoor_air.p)
        m_dot_dry_exhaust = moist_air.mass_flow_from_volume_flow(
            V_dot_exhaust, exhaust_air.T, exhaust_air.x, exhaust_air.p)

        setpoint_T = self.setpoint_T
        setpoint_x = self.setpoint_x

        # Process exhaust side first (adiabatic cooling modifies exhaust state
        # before it enters the heat recovery unit)
        exhaust_air_processed, exhaust_results = self._process_exhaust_components(
            exhaust_air=exhaust_air,
            setpoint_T=setpoint_T,
            setpoint_x=setpoint_x,
            timestep_h=timestep_h,
            m_dot_dry_exhaust=m_dot_dry_exhaust,
            V_dot_supply=V_dot_supply,
            V_dot_exhaust=V_dot_exhaust,
        )

        # Process supply side sequentially
        air = outdoor_air.copy()
        supply_results: dict[str, ComponentResult] = {}

        for idx, comp in enumerate(self.supply_air_components):
            kwargs: dict = {
                "timestep_h": timestep_h,
                "m_dot_dry": m_dot_dry_supply,
            }

            # Heat recovery components need exhaust air info
            if isinstance(comp, _HEAT_RECOVERY_TYPES):
                kwargs["exhaust_air"] = exhaust_air_processed
                kwargs["V_dot_supply"] = V_dot_supply
                kwargs["V_dot_exhaust"] = V_dot_exhaust
                kwargs["T_outdoor"] = outdoor_air.T

            # Heating/cooling coils get downstream fan correction
            effective_setpoint_T = setpoint_T
            if isinstance(comp, _FAN_CORRECTION_TYPES):
                downstream_fan = self._find_downstream_fan(
                    self.supply_air_components, idx)
                dT_fan = self._compute_dT_fan(downstream_fan, m_dot_dry_supply)
                kwargs["dT_fan_correction"] = dT_fan

                # Pre-heating coil: adjust target for downstream humidifier
                if isinstance(comp, HeatingCoil) and comp.coil_type == 'pre' \
                        and air.x < setpoint_x:
                    humidifier = self._find_downstream_humidifier(idx)
                    if isinstance(humidifier, SprayHumidifier):
                        T_pre = self._compute_preheat_target_for_spray(
                            air, setpoint_T, setpoint_x, humidifier, dT_fan)
                        effective_setpoint_T = T_pre + dT_fan  # undo fan correction inside
                    elif isinstance(humidifier, SteamHumidifier):
                        T_pre = self._compute_preheat_target_for_steam(
                            air, setpoint_T, setpoint_x, humidifier, dT_fan)
                        effective_setpoint_T = T_pre + dT_fan

            result = comp.calculate(
                air_in=air,
                setpoint_T=effective_setpoint_T,
                setpoint_x=setpoint_x,
                **kwargs,
            )
            air = result.air_out
            supply_results[comp.name] = result

        # Merge all component results
        all_results = {}
        for name, res in exhaust_results.items():
            all_results[f"exhaust_{name}"] = res
        all_results.update(supply_results)

        # Accumulate totals
        total_Q_heat = sum(r.Q_heat for r in all_results.values())
        total_Q_cool = sum(r.Q_cool for r in all_results.values())
        total_W_el = sum(r.W_el for r in all_results.values())
        total_m_water = sum(r.m_water for r in all_results.values())

        return SystemResult(
            air_supply_out=air,
            Q_heat=total_Q_heat,
            Q_cool=total_Q_cool,
            W_el=total_W_el,
            m_water=total_m_water,
            component_results=all_results,
        )

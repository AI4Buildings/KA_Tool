"""Hourly simulation over weather data for an AHU system."""

from dataclasses import dataclass, field
import pandas as pd

from src.system.ahu_system import AHUSystem, SystemResult
from src.thermodynamics.air_state import AirState
from src.thermodynamics import moist_air


@dataclass
class SimulationResult:
    """Result of a full hourly simulation.

    Attributes:
        hourly_results: List of per-timestep dicts with key quantities
        total_Q_heat_kWh: Total heating energy [kWh]
        total_Q_cool_kWh: Total cooling energy [kWh]
        total_W_el_kWh: Total electrical energy [kWh]
        total_m_water_kg: Total water consumption [kg]
        monthly_breakdown: Monthly energy totals {YYYY-MM: {Q_heat, Q_cool, W_el, m_water}}
        peak_loads: Peak instantaneous loads {Q_heat_peak_kW, Q_cool_peak_kW}
    """
    hourly_results: list[dict]
    total_Q_heat_kWh: float
    total_Q_cool_kWh: float
    total_W_el_kWh: float
    total_m_water_kg: float
    monthly_breakdown: dict[str, dict]
    peak_loads: dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert hourly results to a DataFrame."""
        return pd.DataFrame(self.hourly_results)


class HourlySimulation:
    """Runs an hourly simulation of an AHU system over weather data.

    Args:
        ahu: The AHU system to simulate
        exhaust_air_T: Exhaust/room air temperature [°C]
        exhaust_air_phi: Exhaust/room air relative humidity [fraction 0-1]
    """

    def __init__(self, ahu: AHUSystem,
                 exhaust_air_T: float = 22.0,
                 exhaust_air_phi: float = 0.45):
        self.ahu = ahu
        self.exhaust_air_T = exhaust_air_T
        self.exhaust_air_phi = exhaust_air_phi

    def run(self, weather_data: pd.DataFrame,
            V_dot_supply: float | None = None,
            V_dot_exhaust: float | None = None) -> SimulationResult:
        """Run hourly simulation over weather data.

        Args:
            weather_data: DataFrame with columns 'T' [°C] and 'phi' [fraction 0-1],
                         indexed by DatetimeIndex
            V_dot_supply: Override supply airflow [m³/h], default uses AHU design
            V_dot_exhaust: Override exhaust airflow [m³/h], default uses AHU design

        Returns:
            SimulationResult with hourly and aggregated results
        """
        exhaust_x = moist_air.absolute_humidity(
            self.exhaust_air_phi, self.exhaust_air_T)
        exhaust_air = AirState(T=self.exhaust_air_T, x=exhaust_x)

        hourly_results: list[dict] = []
        monthly_accum: dict[str, dict] = {}

        # Track peaks (instantaneous power = energy / timestep_h, here 1h so kWh = kW)
        peak_Q_heat_kW = 0.0
        peak_Q_cool_kW = 0.0

        total_Q_heat = 0.0
        total_Q_cool = 0.0
        total_W_el = 0.0
        total_m_water = 0.0

        timestep_h = 1.0

        for timestamp, row in weather_data.iterrows():
            T_outdoor = float(row['T'])
            phi_outdoor = float(row['phi'])
            x_outdoor = moist_air.absolute_humidity(phi_outdoor, T_outdoor)
            outdoor_air = AirState(T=T_outdoor, x=x_outdoor)

            result: SystemResult = self.ahu.calculate_timestep(
                outdoor_air=outdoor_air,
                exhaust_air=exhaust_air,
                V_dot_supply=V_dot_supply,
                V_dot_exhaust=V_dot_exhaust,
                timestep_h=timestep_h,
            )

            # Record hourly data
            hourly_results.append({
                'timestamp': timestamp,
                'T_outdoor': T_outdoor,
                'phi_outdoor': phi_outdoor,
                'x_outdoor': x_outdoor,
                'T_supply': result.air_supply_out.T,
                'x_supply': result.air_supply_out.x,
                'phi_supply': result.air_supply_out.phi,
                'Q_heat': result.Q_heat,
                'Q_cool': result.Q_cool,
                'W_el': result.W_el,
                'm_water': result.m_water,
            })

            # Accumulate totals
            total_Q_heat += result.Q_heat
            total_Q_cool += result.Q_cool
            total_W_el += result.W_el
            total_m_water += result.m_water

            # Track peaks (for 1h timestep, kWh equals kW average)
            Q_heat_kW = result.Q_heat / timestep_h
            Q_cool_kW = result.Q_cool / timestep_h
            peak_Q_heat_kW = max(peak_Q_heat_kW, Q_heat_kW)
            peak_Q_cool_kW = max(peak_Q_cool_kW, Q_cool_kW)

            # Monthly accumulation
            month_key = pd.Timestamp(timestamp).strftime('%Y-%m')
            if month_key not in monthly_accum:
                monthly_accum[month_key] = {
                    'Q_heat_kWh': 0.0, 'Q_cool_kWh': 0.0,
                    'W_el_kWh': 0.0, 'm_water_kg': 0.0, 'hours': 0,
                }
            m = monthly_accum[month_key]
            m['Q_heat_kWh'] += result.Q_heat
            m['Q_cool_kWh'] += result.Q_cool
            m['W_el_kWh'] += result.W_el
            m['m_water_kg'] += result.m_water
            m['hours'] += 1

        return SimulationResult(
            hourly_results=hourly_results,
            total_Q_heat_kWh=total_Q_heat,
            total_Q_cool_kWh=total_Q_cool,
            total_W_el_kWh=total_W_el,
            total_m_water_kg=total_m_water,
            monthly_breakdown=monthly_accum,
            peak_loads={
                'Q_heat_peak_kW': peak_Q_heat_kW,
                'Q_cool_peak_kW': peak_Q_cool_kW,
            },
        )

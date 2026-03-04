"""Example: AHU with plate heat recovery (Plattenwärmeübertrager).

Simulates a typical office building AHU with:
- Plate heat exchanger (60% effectiveness)
- Pre-heating coil
- Cooling coil with dehumidification
- Steam humidifier
- Post-heating coil
- Supply fan
"""

from src.thermodynamics.air_state import AirState
from src.components.fan import Fan
from src.components.frost_protection import FrostProtectionPreheater
from src.components.heat_recovery import PlateHeatRecovery
from src.components.heating_coil import HeatingCoil
from src.components.cooling_coil import CoolingCoil
from src.components.humidifier import SteamHumidifier
from src.system.ahu_system import AHUSystem


def create_plate_hrv_system(airflow: float = 5000.0) -> AHUSystem:
    """Create a standard AHU with plate heat recovery."""
    return AHUSystem(
        supply_air_components=[
            FrostProtectionPreheater(limit_temperature=0.0),
            PlateHeatRecovery(
                design_airflow_supply=airflow,
                design_airflow_exhaust=airflow,
                effectiveness_nominal=0.6,
            ),
            HeatingCoil(coil_type='pre', design_capacity_kW=30.0),
            CoolingCoil(dehumidification=True),
            SteamHumidifier(),
            HeatingCoil(coil_type='post'),
            Fan(P_el_kW=1.5, heat_recovery_factor=1.0),
        ],
        exhaust_air_components=[],
        design_airflow_supply=airflow,
        design_airflow_exhaust=airflow,
        setpoint_T=22.0,
        setpoint_x=0.008,
    )


if __name__ == "__main__":
    ahu = create_plate_hrv_system()

    # Test with a few representative conditions
    exhaust = AirState(T=22.0, x=0.008)

    conditions = [
        ("Winter (-12°C)", AirState(T=-12.0, x=0.001)),
        ("Spring (8°C)", AirState(T=8.0, x=0.004)),
        ("Summer (32°C, humid)", AirState(T=32.0, x=0.012)),
        ("Autumn (15°C)", AirState(T=15.0, x=0.007)),
    ]

    print("Plate HRV System - Single Timestep Results")
    print("=" * 70)
    for name, outdoor in conditions:
        result = ahu.calculate_timestep(outdoor_air=outdoor, exhaust_air=exhaust)
        print(f"\n{name}:")
        print(f"  Supply air: T={result.air_supply_out.T:.1f}°C, "
              f"x={result.air_supply_out.x:.4f} kg/kg, "
              f"φ={result.air_supply_out.phi:.0%}")
        print(f"  Q_heat={result.Q_heat:.2f} kWh, "
              f"Q_cool={result.Q_cool:.2f} kWh, "
              f"W_el={result.W_el:.2f} kWh")

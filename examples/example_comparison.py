"""Example: Compare two AHU concepts - plate HRV vs. rotary HRV.

Runs a simulation for both concepts with the same weather data and
compares the total energy demand.

Requires internet access for Open-Meteo weather data.
"""

import pandas as pd

from src.system.simulation import HourlySimulation
from examples.example_plate_hrv import create_plate_hrv_system
from examples.example_rotary_hrv import create_rotary_hrv_system
from src.weather.open_meteo import OpenMeteoClient


def main():
    # Location and period
    location = "Wien"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # Fetch weather data
    client = OpenMeteoClient()
    lat, lon, name = client.geocode(location)
    print(f"Location: {name} ({lat:.2f}, {lon:.2f})")
    print(f"Period: {start_date} to {end_date}")
    print("Fetching weather data...")
    weather = client.get_hourly_weather(lat, lon, start_date, end_date)
    print(f"  {len(weather)} hourly records loaded.")

    # Create both systems
    concepts = {
        "Plate HRV (60%)": create_plate_hrv_system(airflow=5000),
        "Rotary HRV (Sorption)": create_rotary_hrv_system(airflow=5000),
    }

    print("\n" + "=" * 70)
    print("Energy Comparison")
    print("=" * 70)

    results = {}
    for concept_name, ahu in concepts.items():
        sim = HourlySimulation(ahu, exhaust_air_T=22.0, exhaust_air_phi=0.45)
        sim_result = sim.run(weather)
        results[concept_name] = sim_result

        print(f"\n{concept_name}:")
        print(f"  Total heating:    {sim_result.total_Q_heat_kWh:>10.0f} kWh")
        print(f"  Total cooling:    {sim_result.total_Q_cool_kWh:>10.0f} kWh")
        print(f"  Total electrical: {sim_result.total_W_el_kWh:>10.0f} kWh")
        print(f"  Total water:      {sim_result.total_m_water_kg:>10.0f} kg")
        print(f"  Peak heating:     {sim_result.peak_loads['Q_heat_peak_kW']:>10.1f} kW")
        print(f"  Peak cooling:     {sim_result.peak_loads['Q_cool_peak_kW']:>10.1f} kW")


if __name__ == "__main__":
    main()

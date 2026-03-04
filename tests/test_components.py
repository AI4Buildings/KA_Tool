"""Tests for AHU components - energy balance consistency."""

import pytest
from src.thermodynamics.air_state import AirState, ComponentResult
from src.thermodynamics import moist_air
from src.components.fan import Fan
from src.components.frost_protection import FrostProtectionPreheater
from src.components.heating_coil import HeatingCoil
from src.components.cooling_coil import CoolingCoil
from src.components.humidifier import SprayHumidifier, SteamHumidifier
from src.components.heat_recovery import PlateHeatRecovery


class TestFan:
    def test_temperature_rise(self):
        """Fan should raise temperature proportional to power."""
        fan = Fan(P_el_kW=1.5, heat_recovery_factor=1.0)
        air = AirState(T=20.0, x=0.008)
        m_dot = 1.5  # kg/s

        result = fan.calculate(air, setpoint_T=22.0, setpoint_x=0.008,
                               m_dot_dry=m_dot)
        assert result.air_out.T > air.T
        assert result.W_el > 0

    def test_no_power(self):
        """Zero power fan should not change air."""
        fan = Fan(P_el_kW=0.0)
        air = AirState(T=20.0, x=0.008)
        result = fan.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.T == air.T

    def test_temperature_rise_method(self):
        fan = Fan(P_el_kW=2.0, heat_recovery_factor=1.0)
        dT = fan.temperature_rise(m_dot_dry=1.0)
        # dT = P / (m_dot * c_p) = 2.0 / (1.0 * 1.006) ≈ 1.99
        assert abs(dT - 2.0 / 1.006) < 0.01


class TestFrostProtection:
    def test_cold_air_heated(self):
        """Air below limit should be heated to limit."""
        fp = FrostProtectionPreheater(limit_temperature=-5.0)
        air = AirState(T=-15.0, x=0.001)
        result = fp.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert abs(result.air_out.T - (-5.0)) < 0.01
        assert result.W_el > 0

    def test_warm_air_unchanged(self):
        """Air above limit should pass through."""
        fp = FrostProtectionPreheater(limit_temperature=-5.0)
        air = AirState(T=5.0, x=0.003)
        result = fp.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.T == air.T


class TestHeatingCoil:
    def test_heating_needed(self):
        """Cold air should be heated to setpoint."""
        coil = HeatingCoil(coil_type='post')
        air = AirState(T=10.0, x=0.006)
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.008,
                                m_dot_dry=1.0)
        assert abs(result.air_out.T - 22.0) < 0.1
        assert result.Q_heat > 0

    def test_no_heating_needed(self):
        """Warm air should pass through."""
        coil = HeatingCoil(coil_type='post')
        air = AirState(T=25.0, x=0.006)
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.T == air.T
        assert result.Q_heat == 0.0

    def test_capacity_limit(self):
        """Should respect design capacity limit."""
        coil = HeatingCoil(coil_type='pre', design_capacity_kW=5.0)
        air = AirState(T=-20.0, x=0.001)
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=2.0)
        # With limited capacity, may not reach setpoint
        assert result.air_out.T < 22.0
        assert result.Q_heat > 0

    def test_energy_balance(self):
        """Energy should equal m_dot * cp * dT (approximately)."""
        coil = HeatingCoil(coil_type='post')
        air = AirState(T=10.0, x=0.006)
        m_dot = 1.5
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.008,
                                m_dot_dry=m_dot, timestep_h=1.0)
        dT = result.air_out.T - air.T
        Q_expected = m_dot * 1.006 * dT  # kW * 1h = kWh
        assert abs(result.Q_heat - Q_expected) / Q_expected < 0.05


class TestCoolingCoil:
    def test_sensible_cooling(self):
        """Hot air should be cooled to setpoint."""
        coil = CoolingCoil(dehumidification=False)
        air = AirState(T=32.0, x=0.008)
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.010,
                                m_dot_dry=1.0)
        assert abs(result.air_out.T - 22.0) < 0.1
        assert result.Q_cool > 0

    def test_no_cooling_needed(self):
        """Cool air should pass through."""
        coil = CoolingCoil()
        air = AirState(T=18.0, x=0.006)
        result = coil.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.T == air.T


class TestSprayHumidifier:
    def test_humidification(self):
        """Dry air should be humidified."""
        hum = SprayHumidifier(efficiency=0.9)
        air = AirState(T=25.0, x=0.004)
        result = hum.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.x > air.x
        assert result.air_out.T < air.T  # Adiabatic cooling
        assert result.m_water > 0

    def test_already_humid(self):
        """Humid air should pass through."""
        hum = SprayHumidifier()
        air = AirState(T=20.0, x=0.010)
        result = hum.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.x == air.x


class TestSteamHumidifier:
    def test_humidification(self):
        """Dry air should be humidified near-isothermally."""
        hum = SteamHumidifier()
        air = AirState(T=22.0, x=0.004)
        result = hum.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert abs(result.air_out.x - 0.008) < 0.001
        # Temperature should change only slightly
        assert abs(result.air_out.T - air.T) < 3.0
        assert result.W_el > 0

    def test_already_humid(self):
        hum = SteamHumidifier()
        air = AirState(T=22.0, x=0.010)
        result = hum.calculate(air, setpoint_T=22.0, setpoint_x=0.008, m_dot_dry=1.0)
        assert result.air_out.x == air.x


class TestPlateHeatRecovery:
    def test_heating_mode(self):
        """Cold outdoor air should be preheated by exhaust."""
        plate = PlateHeatRecovery(
            design_airflow_supply=5000, design_airflow_exhaust=5000,
            effectiveness_nominal=0.6,
        )
        outdoor = AirState(T=-5.0, x=0.002)
        exhaust = AirState(T=22.0, x=0.008)

        result = plate.calculate(
            outdoor, setpoint_T=20.0, setpoint_x=0.008,
            exhaust_air=exhaust, V_dot_supply=5000, V_dot_exhaust=5000,
        )
        assert result.air_out.T > outdoor.T
        assert result.air_out.T < exhaust.T

    def test_no_exhaust(self):
        """Without exhaust air, output should equal input."""
        plate = PlateHeatRecovery()
        air = AirState(T=5.0, x=0.003)
        result = plate.calculate(air, setpoint_T=20.0, setpoint_x=0.008)
        assert result.air_out.T == air.T

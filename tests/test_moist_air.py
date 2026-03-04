"""Tests for moist air thermodynamic functions.

Validates against known values from h-x diagrams and standard conditions.
"""

import math
import pytest
from src.thermodynamics import moist_air


class TestSaturationPressure:
    def test_zero_degrees(self):
        """At 0°C, p_s ≈ 611 Pa."""
        ps = moist_air.saturation_pressure(0.0)
        assert abs(ps - 611.0) < 5.0

    def test_100_degrees(self):
        """At 100°C, p_s ≈ 101325 Pa."""
        ps = moist_air.saturation_pressure(100.0)
        assert abs(ps - 101325.0) / 101325.0 < 0.05

    def test_20_degrees(self):
        """At 20°C, p_s ≈ 2338 Pa."""
        ps = moist_air.saturation_pressure(20.0)
        assert abs(ps - 2338.0) / 2338.0 < 0.03

    def test_negative(self):
        """At -10°C, p_s ≈ 260 Pa."""
        ps = moist_air.saturation_pressure(-10.0)
        assert abs(ps - 260.0) / 260.0 < 0.05

    def test_monotonically_increasing(self):
        """Saturation pressure should increase with temperature."""
        temps = [-20, -10, 0, 10, 20, 30, 40]
        pressures = [moist_air.saturation_pressure(t) for t in temps]
        for i in range(len(pressures) - 1):
            assert pressures[i] < pressures[i + 1]


class TestSaturationHumidity:
    def test_at_20C(self):
        """At 20°C, x_s ≈ 0.0147 kg/kg."""
        xs = moist_air.saturation_humidity(20.0)
        assert abs(xs - 0.0147) < 0.002

    def test_at_0C(self):
        """At 0°C, x_s ≈ 0.00378 kg/kg."""
        xs = moist_air.saturation_humidity(0.0)
        assert abs(xs - 0.00378) < 0.001


class TestAbsoluteHumidity:
    def test_50_percent_at_20C(self):
        """50% RH at 20°C → x ≈ 0.0073 kg/kg."""
        x = moist_air.absolute_humidity(0.5, 20.0)
        assert abs(x - 0.0073) < 0.001

    def test_100_percent(self):
        """100% RH should give saturation humidity."""
        x = moist_air.absolute_humidity(1.0, 20.0)
        xs = moist_air.saturation_humidity(20.0)
        assert abs(x - xs) / xs < 0.01


class TestRelativeHumidity:
    def test_roundtrip(self):
        """absolute_humidity -> relative_humidity should roundtrip."""
        phi_in = 0.6
        T = 25.0
        x = moist_air.absolute_humidity(phi_in, T)
        phi_out = moist_air.relative_humidity(x, T)
        assert abs(phi_in - phi_out) < 0.001

    def test_saturation(self):
        """At saturation humidity, phi should be ~1.0."""
        xs = moist_air.saturation_humidity(20.0)
        phi = moist_air.relative_humidity(xs, 20.0)
        assert abs(phi - 1.0) < 0.01


class TestEnthalpy:
    def test_dry_air_at_20C(self):
        """Dry air at 20°C: h ≈ 20 kJ/kg."""
        h = moist_air.enthalpy(20.0, 0.0)
        assert abs(h - 20.0) < 1.0

    def test_standard_condition(self):
        """20°C, x=0.0073 (50% RH): h ≈ 38.5 kJ/kg."""
        h = moist_air.enthalpy(20.0, 0.0073)
        assert abs(h - 38.5) < 1.5

    def test_at_zero(self):
        """At 0°C, x=0: h = 0."""
        h = moist_air.enthalpy(0.0, 0.0)
        assert abs(h) < 0.01


class TestTemperatureFromHX:
    def test_roundtrip(self):
        """enthalpy -> temperature_from_h_x should roundtrip."""
        T = 25.0
        x = 0.01
        h = moist_air.enthalpy(T, x)
        T_calc = moist_air.temperature_from_h_x(h, x)
        assert abs(T - T_calc) < 0.01

    def test_dry_air(self):
        h = moist_air.enthalpy(30.0, 0.0)
        T = moist_air.temperature_from_h_x(h, 0.0)
        assert abs(T - 30.0) < 0.01


class TestDewPoint:
    def test_known_dew_point(self):
        """At x=0.0073, dew point ≈ 10°C."""
        T_dew = moist_air.dew_point(0.0073)
        assert abs(T_dew - 10.0) < 2.0

    def test_roundtrip(self):
        """Dew point of saturation humidity at T should be ~T."""
        T = 15.0
        xs = moist_air.saturation_humidity(T)
        T_dew = moist_air.dew_point(xs)
        assert abs(T_dew - T) < 0.5


class TestMixing:
    def test_equal_flows(self):
        """Equal flows should give average."""
        m_dot_mix, T_mix, x_mix, h_mix = moist_air.mixing(1.0, 20.0, 0.006, 1.0, 30.0, 0.010)
        assert abs(x_mix - 0.008) < 0.0001
        # Temperature should be close to average (enthalpy-based mixing)
        assert abs(T_mix - 25.0) < 1.0
        assert abs(m_dot_mix - 2.0) < 0.01

    def test_zero_flow(self):
        """Zero second flow should return first state."""
        m_dot_mix, T_mix, x_mix, h_mix = moist_air.mixing(1.0, 20.0, 0.006, 0.0, 30.0, 0.010)
        assert abs(T_mix - 20.0) < 0.01
        assert abs(x_mix - 0.006) < 0.0001


class TestDensity:
    def test_standard_conditions(self):
        """At 20°C, standard pressure, density ≈ 1.2 kg/m³."""
        rho = moist_air.density_moist_air(20.0, 0.0073)
        assert abs(rho - 1.2) < 0.05

    def test_cold_denser(self):
        """Cold air should be denser than warm air."""
        rho_cold = moist_air.density_moist_air(0.0, 0.003)
        rho_warm = moist_air.density_moist_air(30.0, 0.010)
        assert rho_cold > rho_warm

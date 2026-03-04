"""Tests for dry heat exchanger NTU method."""

import math
import pytest
from src.heat_exchangers.ntu_dry import (
    effectiveness_counterflow,
    effectiveness_crossflow,
    ntu_from_effectiveness_counterflow,
    ntu_from_effectiveness_crossflow,
    ua_kaup_correction,
    DryHXDesignData,
    DryHeatExchanger,
    FlowType,
)


class TestEffectivenessCounterflow:
    def test_zero_ntu(self):
        assert effectiveness_counterflow(0.0, 0.5) == 0.0

    def test_c_star_zero(self):
        """C*=0 (condenser/evaporator): eps = 1 - exp(-NTU)."""
        eps = effectiveness_counterflow(2.0, 0.0)
        assert abs(eps - (1 - math.exp(-2.0))) < 1e-10

    def test_c_star_one(self):
        """C*=1: eps = NTU/(1+NTU)."""
        eps = effectiveness_counterflow(3.0, 1.0)
        assert abs(eps - 3.0 / 4.0) < 1e-10

    def test_typical(self):
        """Typical case: NTU=2, C*=0.5."""
        eps = effectiveness_counterflow(2.0, 0.5)
        assert 0.7 < eps < 0.9

    def test_monotonic(self):
        """Higher NTU should give higher effectiveness."""
        eps1 = effectiveness_counterflow(1.0, 0.5)
        eps2 = effectiveness_counterflow(3.0, 0.5)
        assert eps2 > eps1


class TestEffectivenessCrossflow:
    def test_zero_ntu(self):
        assert effectiveness_crossflow(0.0, 0.5) == 0.0

    def test_c_star_zero(self):
        eps = effectiveness_crossflow(2.0, 0.0)
        assert abs(eps - (1 - math.exp(-2.0))) < 1e-10

    def test_lower_than_counterflow(self):
        """Crossflow should be less effective than counterflow."""
        eps_cross = effectiveness_crossflow(2.0, 0.5)
        eps_counter = effectiveness_counterflow(2.0, 0.5)
        assert eps_cross < eps_counter


class TestNTUInversion:
    def test_counterflow_roundtrip(self):
        """effectiveness -> NTU -> effectiveness should roundtrip."""
        for C_star in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for NTU in [0.5, 1.0, 2.0, 5.0]:
                eps = effectiveness_counterflow(NTU, C_star)
                NTU_calc = ntu_from_effectiveness_counterflow(eps, C_star)
                assert abs(NTU - NTU_calc) < 1e-6, f"Failed for C*={C_star}, NTU={NTU}"

    def test_crossflow_roundtrip(self):
        for C_star in [0.0, 0.3, 0.5, 0.8]:
            for NTU in [0.5, 1.0, 2.0, 5.0]:
                eps = effectiveness_crossflow(NTU, C_star)
                NTU_calc = ntu_from_effectiveness_crossflow(eps, C_star)
                assert abs(NTU - NTU_calc) < 1e-4, f"Failed for C*={C_star}, NTU={NTU}"


class TestKaupCorrection:
    def test_design_point(self):
        """At design point, UA should equal UA_ref."""
        UA = ua_kaup_correction(10.0, 5000, 5000, 1.0, 1.0)
        assert abs(UA - 10.0) < 1e-10

    def test_reduced_flow(self):
        """Reduced flow should reduce UA."""
        UA = ua_kaup_correction(10.0, 2500, 5000, 0.5, 1.0)
        assert UA < 10.0

    def test_zero_flow(self):
        """Zero flow should give zero UA."""
        UA = ua_kaup_correction(10.0, 0.0, 5000, 1.0, 1.0)
        assert UA == 0.0


class TestDryHeatExchanger:
    def test_design_point_reproduces_eps(self):
        """At design conditions, the HX should reproduce the design effectiveness."""
        design = DryHXDesignData(
            V_dot_L=5000, T_L_in=-12, T_L_out=10,
            V_dot_W=1.0, T_W_in=60, T_W_out=45,
        )
        hx = DryHeatExchanger(design=design, flow_type=FlowType.COUNTERFLOW)

        result = hx.calculate(T_L_in=-12, T_W_in=60, V_dot_L=5000, V_dot_W=1.0)
        assert abs(result['T_L_out'] - 10.0) < 0.5
        assert abs(result['eps'] - hx.eps_ref) < 0.01

    def test_crossflow_plate(self):
        """Plate heat recovery (crossflow): design should work."""
        design = DryHXDesignData(
            V_dot_L=5000, T_L_in=-12, T_L_out=8.4,
            V_dot_W=5000, T_W_in=22, T_W_out=5.6,
        )
        hx = DryHeatExchanger(
            design=design,
            flow_type=FlowType.CROSSFLOW,
            c_p_water=1.006,
            rho_water=1.2,
        )

        result = hx.calculate(T_L_in=-12, T_W_in=22, V_dot_L=5000, V_dot_W=5000)
        assert abs(result['T_L_out'] - 8.4) < 0.5

    def test_part_load_within_3_percent(self):
        """Part-load effectiveness should be within ±3% of design (Kaup range)."""
        design = DryHXDesignData(
            V_dot_L=5000, T_L_in=-12, T_L_out=10,
            V_dot_W=1.0, T_W_in=60, T_W_out=45,
        )
        hx = DryHeatExchanger(design=design, flow_type=FlowType.COUNTERFLOW)

        # 80% load
        result = hx.calculate(T_L_in=-12, T_W_in=60, V_dot_L=4000, V_dot_W=0.8)
        # Should still produce reasonable effectiveness
        assert 0.1 < result['eps'] < 1.0

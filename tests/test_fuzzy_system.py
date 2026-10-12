"""
Logic and Unit Tests for Mamdani Fuzzy Inference System.

Note: As specified in project brief Section 8 & Appendix B, these are
computational logic and sanity tests verifying algorithm behavior,
NOT field-trial agronomic validations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fuzzy_system import PearlMilletIrrigationFIS


@pytest.fixture
def fis() -> PearlMilletIrrigationFIS:
    return PearlMilletIrrigationFIS()


def test_membership_functions_range_and_partitions(fis: PearlMilletIrrigationFIS):
    """Test that all membership functions stay bounded in [0, 1]."""
    for sm_val in [0, 20, 35, 50, 70, 95]:
        m = fis.calculate_membership_degrees(sm_val, 30, 50, 10)
        for var_name, linguistic_dict in m.items():
            for label, degree in linguistic_dict.items():
                assert 0.0 <= degree <= 1.0, f"{var_name}.{label} out of [0, 1]: {degree}"
                assert not np.isnan(degree)


def test_scenario_extreme_dry_high_irrigation(fis: PearlMilletIrrigationFIS):
    """Logic Test: Parched soil + high heat + low humidity + no rain must yield High irrigation."""
    res = fis.evaluate(
        soil_moisture=10.0,
        temperature=40.0,
        relative_humidity=20.0,
        recent_rainfall=0.0,
        crop_stage="Flowering",
    )
    assert res.category == "High"
    assert res.irrigation_depth_mm >= 8.0
    assert len(res.active_rules) > 0
    # Confirm Rule 16 or 17 fired
    fired_ids = [r.rule_id for r in res.active_rules]
    assert any(rid in [15, 16, 17] for rid in fired_ids)


def test_scenario_monsoon_wet_zero_irrigation(fis: PearlMilletIrrigationFIS):
    """Logic Test: Saturated soil + recent heavy rain must recommend None/minimal irrigation."""
    res = fis.evaluate(
        soil_moisture=85.0,
        temperature=27.0,
        relative_humidity=85.0,
        recent_rainfall=45.0,
        crop_stage="Flowering",
    )
    assert res.category == "None"
    assert res.irrigation_depth_mm < 1.5
    fired_ids = [r.rule_id for r in res.active_rules]
    assert 1 in fired_ids  # Rule 1: Wet soil or high rain -> None


def test_monotonic_temperature_response(fis: PearlMilletIrrigationFIS):
    """Logic Test: Holding other variables constant, rising temperature should not decrease irrigation."""
    res_cool = fis.evaluate(soil_moisture=45, temperature=22, relative_humidity=45, recent_rainfall=5)
    res_hot = fis.evaluate(soil_moisture=45, temperature=38, relative_humidity=45, recent_rainfall=5)
    assert res_hot.irrigation_depth_mm >= res_cool.irrigation_depth_mm


def test_monotonic_soil_moisture_response(fis: PearlMilletIrrigationFIS):
    """Logic Test: As soil moisture decreases, irrigation recommendation should increase or stay equal."""
    res_wet = fis.evaluate(soil_moisture=75, temperature=32, relative_humidity=50, recent_rainfall=0)
    res_mod = fis.evaluate(soil_moisture=45, temperature=32, relative_humidity=50, recent_rainfall=0)
    res_dry = fis.evaluate(soil_moisture=15, temperature=32, relative_humidity=50, recent_rainfall=0)

    assert res_mod.irrigation_depth_mm >= res_wet.irrigation_depth_mm
    assert res_dry.irrigation_depth_mm >= res_mod.irrigation_depth_mm


def test_out_of_bounds_input_handling(fis: PearlMilletIrrigationFIS):
    """Test that out-of-range inputs trigger warnings and are clamped safely."""
    res = fis.evaluate(
        soil_moisture=-10.0,  # Negative soil moisture
        temperature=55.0,    # Above calibrated max
        relative_humidity=110.0, # Above 100%
        recent_rainfall=-5.0,
    )
    assert len(res.warnings) > 0
    # System should still produce a valid bounded recommendation without crashing
    assert 0.0 <= res.irrigation_depth_mm <= 12.0
    assert res.category in ["None", "Low", "Medium", "High"]


def test_repeatability(fis: PearlMilletIrrigationFIS):
    """Test that identical inputs produce exact identical output deterministically."""
    res1 = fis.evaluate(40.0, 31.0, 50.0, 10.0)
    res2 = fis.evaluate(40.0, 31.0, 50.0, 10.0)
    assert res1.irrigation_depth_mm == res2.irrigation_depth_mm
    assert res1.category == res2.category
    assert len(res1.active_rules) == len(res2.active_rules)


def test_crop_stage_sensitivity(fis: PearlMilletIrrigationFIS):
    """Test that Flowering (peak critical period) demands more irrigation than Maturity."""
    res_flowering = fis.evaluate(35, 34, 40, 2, crop_stage="Flowering")
    res_maturity = fis.evaluate(35, 34, 40, 2, crop_stage="Maturity")
    assert res_flowering.irrigation_depth_mm > res_maturity.irrigation_depth_mm


def test_complete_universe_coverage(fis: PearlMilletIrrigationFIS):
    """Test that every universe of discourse has no gaps (membership sum > 0 everywhere)."""
    # Soil moisture grid
    for sm in np.linspace(0, 100, 101):
        m = fis.calculate_membership_degrees(sm, 30, 50, 10)["soil_moisture"]
        assert sum(m.values()) > 0.0, f"Soil moisture gap at {sm}%"

    # Temperature grid
    for temp in np.linspace(15, 48, 67):
        m = fis.calculate_membership_degrees(50, temp, 50, 10)["temperature"]
        assert sum(m.values()) > 0.0, f"Temperature gap at {temp}C"

    # RH grid
    for rh in np.linspace(10, 100, 91):
        m = fis.calculate_membership_degrees(50, 30, rh, 10)["relative_humidity"]
        assert sum(m.values()) > 0.0, f"RH gap at {rh}%"

    # Rainfall grid
    for rain in np.linspace(0, 60, 61):
        m = fis.calculate_membership_degrees(50, 30, 50, rain)["recent_rainfall"]
        assert sum(m.values()) > 0.0, f"Rainfall gap at {rain}mm"


def test_output_bounds_across_grid(fis: PearlMilletIrrigationFIS):
    """Test that recommendations remain strictly within [0.0, 12.0] mm/day across a grid."""
    for sm in [5, 25, 45, 65, 85]:
        for temp in [20, 32, 42]:
            for rain in [0, 10, 35]:
                res = fis.evaluate(sm, temp, 50, rain, crop_stage="Flowering")
                assert 0.0 <= res.irrigation_depth_mm <= 12.0
                assert res.category in ["None", "Low", "Medium", "High"]


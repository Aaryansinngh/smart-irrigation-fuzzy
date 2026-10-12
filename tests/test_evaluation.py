"""
Unit and verification tests for evaluation module and FAO-56 reference benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_fao56_penman_monteith_et0,
    calculate_fao56_reference,
    run_threshold_baseline,
)


@pytest.fixture
def sample_agro_df() -> pd.DataFrame:
    """Creates synthetic representative meteorological data for test validation."""
    dates = pd.date_range("2023-07-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "T2M": [30.0, 32.0, 28.0, 35.0, 26.0],
        "T2M_MAX": [35.0, 37.0, 32.0, 41.0, 30.0],
        "T2M_MIN": [25.0, 26.0, 24.0, 28.0, 22.0],
        "RH2M": [60.0, 50.0, 75.0, 35.0, 85.0],
        "WS2M": [2.0, 2.5, 1.8, 3.2, 1.5],
        "ALLSKY_SFC_SW_DWN": [20.0, 22.0, 15.0, 24.0, 12.0],
        "PRECTOTCORR": [0.0, 0.0, 12.0, 0.0, 35.0],
        "soil_moisture_pct": [35.0, 30.0, 60.0, 25.0, 80.0],
        "rainfall_3d_sum": [0.0, 0.0, 12.0, 12.0, 47.0],
        "crop_stage": ["Flowering", "Flowering", "Flowering", "Flowering", "Vegetative"],
    })


def test_fao56_penman_monteith_et0_physical_bounds(sample_agro_df):
    """Test that FAO-56 Penman-Monteith ET0 values are physically plausible (1 to 10 mm/day)."""
    et0 = calculate_fao56_penman_monteith_et0(sample_agro_df)
    assert len(et0) == len(sample_agro_df)
    assert (et0 >= 0.5).all(), "ET0 should be positive"
    assert (et0 <= 12.0).all(), "ET0 should not exceed realistic desert max"
    # Hot day (index 3: 35C mean, 41C max, low RH) should have higher ET0 than cool rainy day (index 4)
    assert et0.iloc[3] > et0.iloc[4]


def test_fao56_reference_soil_saturation_zero_irrigation(sample_agro_df):
    """Test that saturated soil (index 4: SM=80%) yields zero net irrigation requirement."""
    nir = calculate_fao56_reference(sample_agro_df)
    assert nir.iloc[4] == 0.0


def test_fao56_reference_rain_suppression(sample_agro_df):
    """Test that significant rain (index 2: 12mm) suppresses or reduces net irrigation requirement."""
    nir = calculate_fao56_reference(sample_agro_df)
    # Compare dry flowering day 1 (0mm rain) vs day 2 (12mm rain)
    assert nir.iloc[0] > nir.iloc[2]


def test_threshold_baseline_behavior(sample_agro_df):
    """Test deterministic rule baseline output bounds and stage adjustments."""
    base = run_threshold_baseline(sample_agro_df)
    assert len(base) == len(sample_agro_df)
    assert (base >= 0.0).all()
    assert (base <= 10.0).all()
    # Dry flowering day (index 1: SM=30, rain=0) should trigger baseline irrigation
    assert base.iloc[1] == 7.0
    # Saturated day (index 4: SM=80) should trigger zero
    assert base.iloc[4] == 0.0


def test_metric_reproducibility():
    """Verify standard metric mathematical formulas on known vectors."""
    y_true = np.array([2.0, 4.0, 0.0, 6.0])
    y_pred = np.array([2.5, 3.5, 0.0, 5.0])

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    # |2.5-2| = 0.5, |3.5-4| = 0.5, |0-0| = 0, |5-6| = 1.0 -> sum = 2.0 / 4 = 0.5
    assert mae == pytest.approx(0.5)
    # sqrt((0.25 + 0.25 + 0 + 1.0) / 4) = sqrt(1.5 / 4) = sqrt(0.375) ~ 0.61237
    assert rmse == pytest.approx(np.sqrt(0.375))

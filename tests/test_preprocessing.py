"""
Unit tests for data preprocessing module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    DataQualityError,
    clean_missing_and_sentinels,
    engineer_agricultural_features,
    validate_physical_bounds,
    validate_schema,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a synthetic sample dataset resembling NASA POWER output."""
    dates = pd.date_range("2023-07-01", periods=10, freq="D")
    data = {
        "date": dates,
        "T2M": [32.0, 31.5, 30.0, 33.5, 34.0, 29.0, 28.5, 31.0, 32.5, 30.0],
        "T2M_MAX": [37.0, 36.0, 35.0, 38.0, 39.0, 33.0, 32.0, 35.0, 37.0, 34.0],
        "T2M_MIN": [26.0, 25.0, 24.0, 27.0, 28.0, 24.0, 23.0, 25.0, 26.0, 24.0],
        "RH2M": [65.0, 70.0, 75.0, 60.0, 55.0, 80.0, 85.0, 65.0, 60.0, 70.0],
        "PRECTOTCORR": [0.0, 5.0, 15.0, 0.0, 0.0, 25.0, 2.0, 0.0, 0.0, 8.0],
        "GWETTOP": [0.35, 0.40, 0.55, 0.48, 0.42, 0.65, 0.60, 0.50, 0.45, 0.52],
        "GWETROOT": [0.40, 0.42, 0.50, 0.48, 0.45, 0.60, 0.58, 0.52, 0.48, 0.51],
        "ALLSKY_SFC_SW_DWN": [20.0, 18.0, 15.0, 22.0, 23.0, 14.0, 16.0, 21.0, 22.0, 19.0],
        "WS2M": [2.1, 2.3, 1.8, 2.5, 3.0, 2.0, 1.7, 2.2, 2.4, 2.0],
    }
    return pd.DataFrame(data)


def test_schema_validation_success(sample_raw_data):
    """Test that schema validation passes when all columns exist."""
    validate_schema(sample_raw_data)


def test_schema_validation_failure():
    """Test that missing column raises DataQualityError."""
    df_bad = pd.DataFrame({"T2M": [25.0], "RH2M": [50.0]})
    with pytest.raises(DataQualityError):
        validate_schema(df_bad)


def test_clean_missing_and_sentinels(sample_raw_data):
    """Test replacement of -999.0 with interpolated or ffilled values."""
    df_with_sentinel = sample_raw_data.copy()
    df_with_sentinel.loc[2, "T2M"] = -999.0
    cleaned = clean_missing_and_sentinels(df_with_sentinel)
    assert cleaned.loc[2, "T2M"] != -999.0
    assert not np.isnan(cleaned.loc[2, "T2M"])
    # Should match previous valid observation (ffill)
    assert cleaned.loc[2, "T2M"] == pytest.approx(31.5)


def test_physical_bounds_check(sample_raw_data):
    """Test bounds checking logic flags impossible values."""
    df_anom = sample_raw_data.copy()
    df_anom.loc[0, "T2M"] = 65.0  # Unrealistic high
    df_anom.loc[1, "RH2M"] = 120.0  # Above 100%
    anomalies = validate_physical_bounds(df_anom)
    assert anomalies["T2M"] == 1
    assert anomalies["RH2M"] == 1
    assert anomalies["GWETROOT"] == 0


def test_agricultural_feature_engineering(sample_raw_data):
    """Test derived feature ranges and rolling calculation logic."""
    df_feat = engineer_agricultural_features(sample_raw_data)

    # Check scaling of soil moisture to 0-100%
    assert (df_feat["soil_moisture_pct"] >= 0.0).all()
    assert (df_feat["soil_moisture_pct"] <= 100.0).all()
    assert df_feat.loc[0, "soil_moisture_pct"] == pytest.approx(40.0)

    # Check 3-day rainfall rolling sum
    # Index 0: 0.0
    # Index 1: 0.0 + 5.0 = 5.0
    # Index 2: 0.0 + 5.0 + 15.0 = 20.0
    assert df_feat.loc[0, "rainfall_3d_sum"] == pytest.approx(0.0)
    assert df_feat.loc[1, "rainfall_3d_sum"] == pytest.approx(5.0)
    assert df_feat.loc[2, "rainfall_3d_sum"] == pytest.approx(20.0)

    # Check Kharif season tagging for July
    assert (df_feat["is_kharif"] == True).all()
    assert set(df_feat["crop_stage"].unique()).issubset({"Vegetative", "Flowering", "Maturity"})

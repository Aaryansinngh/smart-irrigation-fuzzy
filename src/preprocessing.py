"""
Data Preprocessing Pipeline for Smart Irrigation Recommendation System.

Validates schema, parses dates, checks data quality bounds, engineers
agricultural features (rolling rainfall, soil moisture percentage),
and filters for the pearl millet (bajra) Kharif cultivation windows in Jaipur.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "GWETTOP",
    "GWETROOT",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

# Physical plausible bounds for quality checks
PHYSICAL_BOUNDS = {
    "T2M": (-10.0, 55.0),
    "T2M_MAX": (-10.0, 60.0),
    "T2M_MIN": (-15.0, 50.0),
    "RH2M": (0.0, 100.0),
    "PRECTOTCORR": (0.0, 500.0),
    "GWETTOP": (0.0, 1.0),
    "GWETROOT": (0.0, 1.0),
    "ALLSKY_SFC_SW_DWN": (0.0, 40.0),
    "WS2M": (0.0, 35.0),
}


class DataQualityError(ValueError):
    """Raised when data quality validation fails critically."""
    pass


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all required columns are present in DataFrame."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"Missing required columns in dataset: {missing}")
    logger.info("Schema validation passed.")


def clean_missing_and_sentinels(df: pd.DataFrame, sentinel: float = -999.0) -> pd.DataFrame:
    """Replace NASA POWER fill values (-999.0) with NaN and interpolate if isolated."""
    df_clean = df.copy()
    # Count occurrences of sentinel
    sentinel_mask = df_clean == sentinel
    sentinel_counts = sentinel_mask.sum().to_dict()
    for col, count in sentinel_counts.items():
        if count > 0:
            logger.warning(f"Found {count} sentinel values in column {col}. Replacing with NaN.")
            df_clean[col] = df_clean[col].replace(sentinel, np.nan)

    # Check remaining nulls
    null_counts = df_clean[EXPECTED_COLUMNS].isna().sum().to_dict()
    for col, count in null_counts.items():
        if count > 0:
            logger.warning(f"Column {col} has {count} missing values. Performing forward fill.")
            df_clean[col] = df_clean[col].ffill().bfill()

    return df_clean


def validate_physical_bounds(df: pd.DataFrame) -> Dict[str, int]:
    """Check for out-of-range physical values. Returns count of anomalies per column."""
    anomalies = {}
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            out_of_bounds = ((df[col] < low) | (df[col] > high)).sum()
            anomalies[col] = int(out_of_bounds)
            if out_of_bounds > 0:
                logger.warning(
                    f"Column '{col}' has {out_of_bounds} values outside expected bounds [{low}, {high}]."
                )
    return anomalies


def engineer_agricultural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling rainfall windows, moisture percentages, and crop stage proxies."""
    df_feat = df.copy()

    # Root-zone and topsoil moisture scaled to 0-100%
    # NASA GWETROOT is fraction (0.0 - 1.0) of soil wetness
    df_feat["soil_moisture_pct"] = (df_feat["GWETROOT"] * 100.0).clip(0.0, 100.0).round(2)
    df_feat["surface_moisture_pct"] = (df_feat["GWETTOP"] * 100.0).clip(0.0, 100.0).round(2)

    # 3-day and 7-day cumulative rainfall (antecedent moisture)
    df_feat["rainfall_3d_sum"] = (
        df_feat["PRECTOTCORR"].rolling(window=3, min_periods=1).sum().round(2)
    )
    df_feat["rainfall_7d_sum"] = (
        df_feat["PRECTOTCORR"].rolling(window=7, min_periods=1).sum().round(2)
    )

    # Daily diurnal temperature range
    df_feat["temp_range"] = (df_feat["T2M_MAX"] - df_feat["T2M_MIN"]).round(2)

    # Kharif cropping season indicator for Pearl Millet in Jaipur:
    # Sowing: mid-June (Day of Year ~166) through harvest in mid-October (Day of Year ~288)
    # Months 6 (June), 7 (July), 8 (August), 9 (September), 10 (October)
    month = df_feat["date"].dt.month
    day = df_feat["date"].dt.day
    is_kharif = (
        ((month == 6) & (day >= 15)) |
        (month.isin([7, 8, 9])) |
        ((month == 10) & (day <= 15))
    )
    df_feat["is_kharif"] = is_kharif

    # Crop stage proxy within each Kharif year:
    # Stage 1: Vegetative / Tillering (approx June 15 - July 25)
    # Stage 2: Booting / Flowering / Grain Filling (approx July 26 - Sept 15) - MOST SENSITIVE
    # Stage 3: Maturity / Ripening (approx Sept 16 - Oct 15)
    def determine_stage(row: pd.Series) -> str:
        if not row["is_kharif"]:
            return "Off-Season"
        m = row["date"].month
        d = row["date"].day
        if (m == 6) or (m == 7 and d <= 25):
            return "Vegetative"
        elif (m == 7 and d > 25) or (m == 8) or (m == 9 and d <= 15):
            return "Flowering"
        else:
            return "Maturity"

    df_feat["crop_stage"] = df_feat.apply(determine_stage, axis=1)

    return df_feat


def run_preprocessing_pipeline(
    raw_csv_path: Path,
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end preprocessing pipeline.
    
    Returns (full_preprocessed_df, kharif_only_df).
    """
    logger.info(f"Loading raw dataset from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)

    # Parse dates
    # raw 'date' column contains integers like 20210601
    df_raw["date"] = pd.to_datetime(df_raw["date"].astype(str), format="%Y%m%d")
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Duplicate check
    duplicates = df_raw.duplicated(subset=["date"]).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate dates. Removing duplicates.")
        df_raw = df_raw.drop_duplicates(subset=["date"], keep="first")

    # Validate Schema
    validate_schema(df_raw)

    # Clean missing / sentinels
    df_clean = clean_missing_and_sentinels(df_raw)

    # Physical bounds check
    anomalies = validate_physical_bounds(df_clean)
    logger.info(f"Physical bounds check summary: {anomalies}")

    # Feature engineering
    df_featured = engineer_agricultural_features(df_clean)

    # Filter Kharif dataset
    df_kharif = df_featured[df_featured["is_kharif"]].copy().reset_index(drop=True)

    # Save to processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    full_path = processed_dir / "jaipur_daily_preprocessed.csv"
    kharif_path = processed_dir / "jaipur_kharif_pearl_millet.csv"

    df_featured.to_csv(full_path, index=False)
    df_kharif.to_csv(kharif_path, index=False)

    logger.info(f"Full preprocessed dataset saved to {full_path} ({len(df_featured)} records)")
    logger.info(f"Kharif pearl millet dataset saved to {kharif_path} ({len(df_kharif)} records)")

    return df_featured, df_kharif


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_csv = base_dir / "data" / "raw" / "nasa_power_jaipur_raw.csv"
    proc_dir = base_dir / "data" / "processed"
    run_preprocessing_pipeline(raw_csv, proc_dir)

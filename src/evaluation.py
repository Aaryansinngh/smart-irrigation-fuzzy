"""
Evaluation and Validation Module for Smart Irrigation FIS.

Evaluates the Mamdani Fuzzy Inference System against:
1. Authentic FAO-56 Penman-Monteith Reference Evapotranspiration (ET0) & Crop Water Requirement (ETc).
2. A deterministic threshold-based baseline model.
3. Sensitivity analysis under continuous input sweeps.

Outputs verified empirical metrics and diagnostic plots on real Jaipur historical data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.fuzzy_system import PearlMilletIrrigationFIS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Geographic constants for Jaipur District
JAIPUR_LAT_DEG = 26.9124
JAIPUR_ELEVATION_M = 435.0  # meters above sea level


def calculate_fao56_penman_monteith_et0(
    df: pd.DataFrame,
    latitude_deg: float = JAIPUR_LAT_DEG,
    elevation_m: float = JAIPUR_ELEVATION_M,
) -> pd.Series:
    """Computes daily FAO-56 Penman-Monteith Reference Evapotranspiration (ET0).

    Strictly implements the FAO Irrigation and Drainage Paper No. 56
    standard physical equation (Allen et al., 1998, Chapter 3).

    Required DataFrame columns:
    - date: datetime or ISO date string
    - T2M: mean daily temperature at 2m (°C)
    - T2M_MAX: maximum daily temperature at 2m (°C)
    - T2M_MIN: minimum daily temperature at 2m (°C)
    - RH2M: mean daily relative humidity at 2m (%)
    - WS2M: wind speed at 2m (m/s)
    - ALLSKY_SFC_SW_DWN: solar radiation incident on horizontal surface (MJ/m^2/day)
    """
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear.values
    lat_rad = np.radians(latitude_deg)

    # 1. Atmospheric pressure P (kPa) and Psychrometric constant gamma (kPa/°C)
    p_atm = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * p_atm

    # 2. Temperature statistics
    t_mean = df["T2M"].values
    t_max = df["T2M_MAX"].values
    t_min = df["T2M_MIN"].values

    # 3. Saturation vapor pressure es (kPa)
    e0_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    e0_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    es = (e0_tmax + e0_tmin) / 2.0

    # 4. Actual vapor pressure ea (kPa) from relative humidity
    rh = df["RH2M"].values
    ea = es * (rh / 100.0)

    # 5. Slope of saturation vapor pressure curve delta (kPa/°C)
    delta = (4098.0 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3)))) / ((t_mean + 237.3) ** 2)

    # 6. Extraterrestrial radiation Ra (MJ/m^2/day)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    solar_dec = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    omega_s = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_dec), -1.0, 1.0))
    g_sc = 0.0820  # MJ/m^2/min solar constant
    ra = (24.0 * 60.0 / np.pi) * g_sc * dr * (
        omega_s * np.sin(lat_rad) * np.sin(solar_dec)
        + np.cos(lat_rad) * np.cos(solar_dec) * np.sin(omega_s)
    )

    # 7. Clear-sky radiation Rso (MJ/m^2/day)
    rso = (0.75 + 2e-5 * elevation_m) * ra

    # 8. Net shortwave radiation Rns (MJ/m^2/day) for grass reference (albedo alpha = 0.23)
    rs = df["ALLSKY_SFC_SW_DWN"].values
    rns = (1.0 - 0.23) * rs

    # 9. Net longwave radiation Rnl (MJ/m^2/day)
    sigma = 4.903e-9  # Stefan-Boltzmann constant (MJ/K^4/m^2/day)
    t_max_k4 = (t_max + 273.16) ** 4
    t_min_k4 = (t_min + 273.16) ** 4
    f_cloud = np.clip(1.35 * (rs / np.maximum(rso, 0.01)) - 0.35, 0.05, 1.0)
    rnl = sigma * ((t_max_k4 + t_min_k4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * f_cloud

    # 10. Net radiation Rn (MJ/m^2/day)
    rn = rns - rnl

    # 11. Wind speed u2 (m/s)
    u2 = np.maximum(0.1, df["WS2M"].values)

    # 12. FAO-56 Penman-Monteith ET0 Equation
    num = 0.408 * delta * rn + gamma * (900.0 / (t_mean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    et0 = np.clip(num / den, 0.5, 12.0)

    return pd.Series(np.round(et0, 2), index=df.index, name="fao56_et0_mm")


def calculate_fao56_reference(df: pd.DataFrame) -> pd.Series:
    """Computes daily FAO-56 Crop Water Requirement (ETc) and Net Irrigation Requirement (NIR)

    for Pearl Millet in Jaipur.

    Kc stages for Pearl Millet (FAO-56 Table 12):
    - Initial / Vegetative: 0.55
    - Mid-season / Flowering: 1.15 (water-critical period)
    - Late-season / Maturity: 0.45
    """
    et0 = calculate_fao56_penman_monteith_et0(df)

    def get_kc(stage: str) -> float:
        if stage == "Vegetative":
            return 0.55
        elif stage == "Flowering":
            return 1.15
        elif stage == "Maturity":
            return 0.45
        return 0.0

    kc = df["crop_stage"].apply(get_kc)
    etc = et0 * kc

    # Effective Rainfall (USDA SCS daily approximation):
    # Rainfall < 5 mm in semi-arid summer mostly evaporates before infiltrating root zone.
    # Rainfall >= 5 mm provides approx 80% effective infiltration.
    daily_precip = df["PRECTOTCORR"]
    peff = np.where(daily_precip >= 5.0, daily_precip * 0.80, 0.0)

    # Net Irrigation Deficit
    nir = np.maximum(0.0, etc - peff)

    # Agronomic Soil Moisture Constraint:
    # If root-zone moisture is at or above field capacity (>65% wetness),
    # existing soil water storage fully meets the transpiration demand, so NIR = 0.
    sm = df["soil_moisture_pct"]
    nir = np.where(sm >= 65.0, 0.0, nir)

    # Bounded to maximum daily irrigation application capacity (0 - 12 mm/day)
    nir = np.clip(nir, 0.0, 12.0)
    return pd.Series(np.round(nir, 2), index=df.index, name="fao56_ref_irrigation_mm")


def run_threshold_baseline(df: pd.DataFrame) -> pd.Series:
    """Standard fixed-threshold scheduler for benchmark comparison."""
    baseline = []
    for _, row in df.iterrows():
        sm = row["soil_moisture_pct"]
        rain = row["rainfall_3d_sum"]
        stage = row["crop_stage"]

        if stage == "Off-Season":
            baseline.append(0.0)
            continue

        if sm < 32.0 and rain < 5.0:
            val = 7.0 if stage == "Flowering" else 5.0
        elif sm < 45.0 and rain < 2.0:
            val = 4.5 if stage == "Flowering" else 3.0
        elif sm < 55.0 and rain == 0.0 and row["T2M"] > 34.0:
            val = 3.0
        else:
            val = 0.0
        baseline.append(val)

    return pd.Series(baseline, index=df.index, name="threshold_baseline_mm")


def evaluate_historical_dataset(
    processed_csv_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs FIS, authentic FAO-56 Penman-Monteith reference, and Baseline across real Kharif records."""
    logger.info(f"Loading preprocessed Kharif dataset from {processed_csv_path}...")
    df = pd.read_csv(processed_csv_path)
    df["date"] = pd.to_datetime(df["date"])

    fis = PearlMilletIrrigationFIS()

    # Run FIS evaluation row-by-row
    fis_recommendations = []
    fis_categories = []
    fired_rule_counts = []

    for _, row in df.iterrows():
        res = fis.evaluate(
            soil_moisture=row["soil_moisture_pct"],
            temperature=row["T2M"],
            relative_humidity=row["RH2M"],
            recent_rainfall=row["rainfall_3d_sum"],
            crop_stage=row["crop_stage"],
        )
        fis_recommendations.append(res.irrigation_depth_mm)
        fis_categories.append(res.category)
        fired_rule_counts.append(len(res.active_rules))

    df["fis_irrigation_mm"] = fis_recommendations
    df["fis_category"] = fis_categories
    df["active_rules_count"] = fired_rule_counts

    # Calculate authentic FAO-56 Reference and Threshold Baseline
    df["fao56_ref_irrigation_mm"] = calculate_fao56_reference(df)
    df["threshold_baseline_mm"] = run_threshold_baseline(df)

    # Quantitative Comparison Metrics
    y_fis = df["fis_irrigation_mm"].values
    y_fao = df["fao56_ref_irrigation_mm"].values
    y_base = df["threshold_baseline_mm"].values

    mae_fis_fao = float(np.mean(np.abs(y_fis - y_fao)))
    rmse_fis_fao = float(np.sqrt(np.mean((y_fis - y_fao) ** 2)))
    corr_fis_fao, _ = stats.pearsonr(y_fis, y_fao)

    mae_base_fao = float(np.mean(np.abs(y_base - y_fao)))
    rmse_base_fao = float(np.sqrt(np.mean((y_base - y_fao) ** 2)))
    corr_base_fao, _ = stats.pearsonr(y_base, y_fao)

    # Binary Agreement on Irrigation Trigger (threshold >= 1.0 mm)
    trigger_fis = y_fis >= 1.0
    trigger_fao = y_fao >= 1.0
    trigger_agreement_rate = float(np.mean(trigger_fis == trigger_fao))

    # Category frequencies
    cat_counts = {str(k): int(v) for k, v in df["fis_category"].value_counts().items()}

    metrics = {
        "dataset_records_count": int(len(df)),
        "kharif_seasons": sorted([int(y) for y in df["date"].dt.year.unique()]),
        "reference_method": "FAO-56 Penman-Monteith (Allen et al. 1998)",
        "fis_vs_fao56": {
            "MAE_mm": round(float(mae_fis_fao), 3),
            "RMSE_mm": round(float(rmse_fis_fao), 3),
            "Pearson_Correlation": round(float(corr_fis_fao), 3),
            "Trigger_Agreement_Rate": round(float(trigger_agreement_rate), 3),
        },
        "baseline_vs_fao56": {
            "MAE_mm": round(float(mae_base_fao), 3),
            "RMSE_mm": round(float(rmse_base_fao), 3),
            "Pearson_Correlation": round(float(corr_base_fao), 3),
        },
        "category_distribution": cat_counts,
        "mean_active_rules_per_day": round(float(df["active_rules_count"].mean()), 2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics written to {metrics_path}")

    # Generate Comparative Time Series Plot for Sample Season (Kharif 2023)
    season_2023 = df[df["date"].dt.year == 2023].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Weather Drivers
    ax1 = axes[0]
    ax1.plot(season_2023["date"], season_2023["T2M"], color="crimson", label="Mean Temp (°C)")
    ax1.plot(season_2023["date"], season_2023["soil_moisture_pct"], color="green", label="Root Soil Moisture (%)")
    ax1.set_ylabel("Temp (°C) / Moisture (%)")
    ax1.set_title("Jaipur Kharif 2023: Environmental Observations & Antecedent Rainfall")
    ax1.legend(loc="upper left")
    ax1_twin = ax1.twinx()
    ax1_twin.bar(season_2023["date"], season_2023["PRECTOTCORR"], color="dodgerblue", alpha=0.4, label="Rainfall (mm)")
    ax1_twin.set_ylabel("Daily Rainfall (mm)")
    ax1_twin.legend(loc="upper right")

    # Subplot 2: Fuzzy vs FAO-56 vs Baseline Recommendations
    ax2 = axes[1]
    ax2.plot(season_2023["date"], season_2023["fis_irrigation_mm"], color="darkblue", lw=2, label="Mamdani FIS Recommendation (mm)")
    ax2.plot(season_2023["date"], season_2023["fao56_ref_irrigation_mm"], color="darkorange", lw=1.5, linestyle="--", label="FAO-56 Penman-Monteith Ref (mm)")
    ax2.plot(season_2023["date"], season_2023["threshold_baseline_mm"], color="gray", lw=1.2, linestyle=":", label="Fixed Threshold Baseline (mm)")
    ax2.set_ylabel("Irrigation (mm/day)")
    ax2.set_title("Irrigation Comparison: Mamdani FIS vs FAO-56 Penman-Monteith Reference Benchmark")
    ax2.legend(loc="upper right")

    # Subplot 3: Active Rules Count
    ax3 = axes[2]
    ax3.fill_between(season_2023["date"], season_2023["active_rules_count"], color="teal", alpha=0.3, label="Active Rules Count")
    ax3.plot(season_2023["date"], season_2023["active_rules_count"], color="teal", lw=1.5)
    ax3.set_ylabel("Active Rules")
    ax3.set_xlabel("Date (Kharif 2023)")
    ax3.set_title("Interpretability: Number of Concurrent Active Mamdani Rules")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plot_path = output_dir / "evaluation_timeseries_kharif2023.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Historical comparison plot saved to {plot_path}")

    # Generate Sensitivity Analysis Surface
    generate_sensitivity_surface(fis, output_dir)

    # Save enriched dataframe for dashboard
    df.to_csv(output_dir / "jaipur_evaluated_historical.csv", index=False)

    return metrics


def generate_sensitivity_surface(fis: PearlMilletIrrigationFIS, output_dir: Path):
    """Computes and visualizes response surface across Soil Moisture and Temperature."""
    sm_range = np.linspace(5, 95, 30)
    temp_range = np.linspace(18, 45, 30)
    sm_grid, temp_grid = np.meshgrid(sm_range, temp_range)
    irr_surface = np.zeros_like(sm_grid)

    for i in range(len(temp_range)):
        for j in range(len(sm_range)):
            res = fis.evaluate(
                soil_moisture=sm_grid[i, j],
                temperature=temp_grid[i, j],
                relative_humidity=50.0,
                recent_rainfall=2.0,
                crop_stage="Flowering",
            )
            irr_surface[i, j] = res.irrigation_depth_mm

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(sm_grid, temp_grid, irr_surface, cmap="viridis", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Soil Moisture (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_zlabel("Irrigation (mm/day)")
    ax.set_title("FIS Sensitivity Surface: Response to Soil Moisture & Temperature (RH=50%, Rain=2mm)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Irrigation Depth (mm/day)")

    plt.tight_layout()
    surf_path = output_dir / "sensitivity_surface.png"
    fig.savefig(surf_path, dpi=200)
    plt.close(fig)
    logger.info(f"Sensitivity surface plot saved to {surf_path}")


if __name__ == "__main__":
    proc_csv = PROJECT_ROOT / "data" / "processed" / "jaipur_kharif_pearl_millet.csv"
    out_dir = PROJECT_ROOT / "data" / "processed"
    results = evaluate_historical_dataset(proc_csv, out_dir)
    print("\n--- Verified Evaluation Summary ---")
    print(json.dumps(results, indent=2))

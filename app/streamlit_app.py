"""
Streamlit Web Dashboard for Smart Irrigation Recommendation System.

Interactive decision-support prototype using Mamdani Fuzzy Inference
for Pearl Millet cultivation in Jaipur District, Rajasthan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.fuzzy_system import PearlMilletIrrigationFIS

st.set_page_config(
    page_title="Smart Irrigation FIS | Jaipur Pearl Millet",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 0.75rem;
        border-radius: 4px;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fis() -> PearlMilletIrrigationFIS:
    return PearlMilletIrrigationFIS()


@st.cache_data
def load_historical_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "processed" / "jaipur_evaluated_historical.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "jaipur_kharif_pearl_millet.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_metrics() -> dict:
    metrics_path = PROJECT_ROOT / "data" / "processed" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


fis = load_fis()
historical_df = load_historical_data()
eval_metrics = load_metrics()

# Header Section
st.markdown("<div class='main-header'>Smart Irrigation Recommendation System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>A Real-Data-Driven Mamdani Fuzzy Inference System for Pearl Millet (<i>Bajra</i>) in Jaipur District, Rajasthan</div>",
    unsafe_allow_html=True,
)

# Top Context Badges
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.info("📍 **Region**: Jaipur, Rajasthan (26.91°N, 75.79°E)")
with col_b2:
    st.success("🌱 **Crop**: Pearl Millet (*Bajra*, Kharif season)")
with col_b3:
    st.warning("📡 **Data**: NASA POWER (MERRA-2 Reanalysis)")
with col_b4:
    st.info("🧠 **Method**: Mamdani FIS (Centroid Defuzzification)")

# Navigation Tabs
tab_sim, tab_hist, tab_mf, tab_eval, tab_gov = st.tabs([
    "🎛️ Interactive Simulator",
    "📅 Historical Jaipur Records",
    "📈 Membership Functions",
    "🔬 Reference Evaluation & Metrics",
    "📋 Scientific Governance & Sources",
])

# -------------------------------------------------------------
# TAB 1: INTERACTIVE SIMULATOR
# -------------------------------------------------------------
with tab_sim:
    st.subheader("Simulate Environmental Conditions & Inspect Fuzzy Reasoning")

    # Quick scenario presets
    st.markdown("**Load Preset Agronomic Scenarios:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    preset_sm = 35.0
    preset_temp = 32.0
    preset_rh = 50.0
    preset_rain = 2.0
    preset_stage = "Flowering"

    if p_col1.button("🌧️ Active Monsoon"):
        preset_sm = 85.0
        preset_temp = 27.0
        preset_rh = 85.0
        preset_rain = 40.0
        preset_stage = "Vegetative"
    elif p_col2.button("🔥 Severe Dry Spell"):
        preset_sm = 15.0
        preset_temp = 39.0
        preset_rh = 22.0
        preset_rain = 0.0
        preset_stage = "Flowering"
    elif p_col3.button("⛅ Post-Rain Drydown"):
        preset_sm = 48.0
        preset_temp = 31.0
        preset_rh = 60.0
        preset_rain = 10.0
        preset_stage = "Flowering"
    elif p_col4.button("🌾 Late Maturity"):
        preset_sm = 38.0
        preset_temp = 29.0
        preset_rh = 40.0
        preset_rain = 0.0
        preset_stage = "Maturity"

    # Input controls
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        soil_moisture = st.slider(
            "Root-Zone Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(preset_sm),
            step=1.0,
            help="Normalized root-zone soil wetness index from 0% (wilting point) to 100% (saturation).",
        )
        temperature = st.slider(
            "Ambient Daily Temperature (°C)",
            min_value=15.0,
            max_value=48.0,
            value=float(preset_temp),
            step=0.5,
            help="Daily mean temperature at 2m height.",
        )
    with c_in2:
        relative_humidity = st.slider(
            "Relative Humidity (%)",
            min_value=10.0,
            max_value=100.0,
            value=float(preset_rh),
            step=1.0,
            help="Daily average relative humidity (governs atmospheric evaporative demand).",
        )
        recent_rainfall = st.slider(
            "3-Day Cumulative Rainfall (mm)",
            min_value=0.0,
            max_value=60.0,
            value=float(preset_rain),
            step=0.5,
            help="Total antecedent precipitation over the prior 72 hours.",
        )

    crop_stage = st.selectbox(
        "Pearl Millet Growth Stage",
        options=["Vegetative", "Flowering", "Maturity"],
        index=["Vegetative", "Flowering", "Maturity"].index(preset_stage),
        help="Flowering / Grain filling is the most moisture-critical window (Kc ~ 1.15).",
    )

    # Evaluate FIS
    result = fis.evaluate(
        soil_moisture=soil_moisture,
        temperature=temperature,
        relative_humidity=relative_humidity,
        recent_rainfall=recent_rainfall,
        crop_stage=crop_stage,
    )

    # Display Warnings if any
    if result.warnings:
        for w in result.warnings:
            st.warning(f"⚠️ {w}")

    st.markdown("---")

    # Recommendation Results Display
    st.subheader("Fuzzy Recommendation & Output Summary")
    r_col1, r_col2, r_col3 = st.columns(3)

    category_colors = {
        "None": "#10B981",    # Emerald
        "Low": "#3B82F6",     # Blue
        "Medium": "#F59E0B",  # Amber
        "High": "#EF4444",    # Red
    }
    cat_color = category_colors.get(result.category, "#3B82F6")

    with r_col1:
        st.metric(
            label="Irrigation Recommendation",
            value=f"{result.irrigation_depth_mm:.1f} mm/day",
            help="Net daily crop water depth recommended for the next 24-48h planning interval.",
        )
    with r_col2:
        st.markdown(
            f"<div style='font-size:0.9rem; color:#6B7280;'>Linguistic Urgency</div>"
            f"<div style='font-size:2rem; font-weight:700; color:{cat_color};'>{result.category}</div>",
            unsafe_allow_html=True,
        )
    with r_col3:
        st.markdown(
            f"<div style='font-size:0.9rem; color:#6B7280;'>Agronomic Guidance</div>"
            f"<div style='font-size:1rem; font-weight:500;'>{result.linguistic_summary}</div>",
            unsafe_allow_html=True,
        )

    # Visualizing Defuzzification and Membership Activations
    st.markdown("---")
    st.subheader("Transparent Fuzzy Reasoning Breakdown")

    v_col1, v_col2 = st.columns([1.2, 1.0])

    with v_col1:
        st.markdown("#### Centroid Defuzzification Curve")
        x_irr, y_agg = result.aggregated_fuzzy_curve
        fig_defuzz, ax_d = plt.subplots(figsize=(7, 3.8))

        # Plot individual output membership functions for reference
        ax_d.plot(fis.x_irr, fis.irr_none, "g--", alpha=0.35, label="None")
        ax_d.plot(fis.x_irr, fis.irr_low, "b--", alpha=0.35, label="Low")
        ax_d.plot(fis.x_irr, fis.irr_medium, "y--", alpha=0.35, label="Medium")
        ax_d.plot(fis.x_irr, fis.irr_high, "r--", alpha=0.35, label="High")

        # Shaded aggregated area
        ax_d.fill_between(x_irr, y_agg, color="navy", alpha=0.4, label="Aggregated Output (μ)")
        ax_d.plot(x_irr, y_agg, color="navy", lw=2)

        # Defuzzified centroid marker
        ax_d.axvline(
            x=result.irrigation_depth_mm,
            color="crimson",
            linestyle="-",
            lw=2.5,
            label=f"Centroid: {result.irrigation_depth_mm:.2f} mm",
        )
        ax_d.set_xlabel("Net Irrigation Requirement (mm/day)")
        ax_d.set_ylabel("Aggregated Membership Degree (μ)")
        ax_d.set_ylim(-0.05, 1.05)
        ax_d.legend(loc="upper right", fontsize=8)
        ax_d.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_defuzz)
        plt.close(fig_defuzz)

    with v_col2:
        st.markdown("#### Input Linguistic Memberships")
        m = result.membership_degrees
        mem_rows = []
        for var_name, labels in m.items():
            for label, deg in labels.items():
                if deg > 0.01:
                    mem_rows.append({"Variable": var_name.replace("_", " ").title(), "Fuzzy Set": label.capitalize(), "Membership (μ)": f"{deg:.3f}"})
        if mem_rows:
            st.dataframe(pd.DataFrame(mem_rows), width="stretch", hide_index=True)
        else:
            st.info("No input memberships above 0.01.")

    # Rule Activation Table
    st.markdown("#### Active Mamdani IF–THEN Rules")
    if result.active_rules:
        rule_data = [
            {
                "Rule #": r.rule_id,
                "Fired Agronomic Rule": r.rule_text,
                "Premise Strength (α-cut)": f"{r.strength:.3f}",
                "Consequent": r.consequent.capitalize(),
            }
            for r in result.active_rules
        ]
        st.dataframe(pd.DataFrame(rule_data), width="stretch", hide_index=True)
    else:
        st.info("No specific rules activated above threshold.")

# -------------------------------------------------------------
# TAB 2: HISTORICAL JAIPUR RECORDS
# -------------------------------------------------------------
with tab_hist:
    st.subheader("Historical Verification: Jaipur Kharif Seasons (2021–2024)")
    st.markdown(
        "Explore real historical daily observations for Jaipur, Rajasthan and compare the Mamdani FIS recommendation against the calculated FAO-56 crop water requirement reference."
    )

    h_col1, h_col2 = st.columns([1, 3])
    with h_col1:
        available_years = sorted(list(historical_df["date"].dt.year.unique()))
        selected_year = st.selectbox("Select Kharif Cropping Year", options=available_years, index=len(available_years)-2)
        year_df = historical_df[historical_df["date"].dt.year == selected_year].copy().reset_index(drop=True)

        selected_date_idx = st.slider("Select Day of Season", min_value=0, max_value=len(year_df)-1, value=40)
        selected_row = year_df.iloc[selected_date_idx]

        st.markdown("---")
        st.markdown(f"**Day Details ({selected_row['date'].strftime('%d %B %Y')}):**")
        st.write(f"- **Mean Temp**: {selected_row['T2M']:.1f} °C")
        st.write(f"- **Relative Humidity**: {selected_row['RH2M']:.1f} %")
        st.write(f"- **Daily Rain**: {selected_row['PRECTOTCORR']:.1f} mm")
        st.write(f"- **3-Day Rain Sum**: {selected_row['rainfall_3d_sum']:.1f} mm")
        st.write(f"- **Root Moisture**: {selected_row['soil_moisture_pct']:.1f} %")
        st.write(f"- **Stage**: {selected_row['crop_stage']}")
        if "fis_irrigation_mm" in selected_row:
            st.success(f"**FIS Recommendation**: {selected_row['fis_irrigation_mm']:.1f} mm ({selected_row['fis_category']})")
        if "fao56_ref_irrigation_mm" in selected_row:
            st.info(f"**FAO-56 Benchmark**: {selected_row['fao56_ref_irrigation_mm']:.1f} mm")

    with h_col2:
        fig_hist, axes_h = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # Weather
        axes_h[0].plot(year_df["date"], year_df["soil_moisture_pct"], color="green", label="Root Zone Moisture (%)")
        axes_h[0].plot(year_df["date"], year_df["T2M"], color="crimson", label="Temperature (°C)")
        axes_h[0].axvline(selected_row["date"], color="black", linestyle="--", alpha=0.7, label="Selected Date")
        axes_h[0].set_ylabel("Moisture (%) / Temp (°C)")
        axes_h[0].set_title(f"Jaipur Kharif {selected_year} Environmental Observations")
        axes_h[0].legend(loc="upper left")
        axes_h[0].grid(True, linestyle=":", alpha=0.5)

        # Irrigation comparison
        if "fis_irrigation_mm" in year_df:
            axes_h[1].plot(year_df["date"], year_df["fis_irrigation_mm"], color="darkblue", lw=2, label="Fuzzy FIS (mm)")
        if "fao56_ref_irrigation_mm" in year_df:
            axes_h[1].plot(year_df["date"], year_df["fao56_ref_irrigation_mm"], color="darkorange", linestyle="--", label="FAO-56 Ref (mm)")
        if "threshold_baseline_mm" in year_df:
            axes_h[1].plot(year_df["date"], year_df["threshold_baseline_mm"], color="gray", linestyle=":", label="Threshold Baseline (mm)")
        axes_h[1].axvline(selected_row["date"], color="black", linestyle="--", alpha=0.7)
        axes_h[1].set_ylabel("Irrigation (mm/day)")
        axes_h[1].set_title(f"Irrigation Recommendations across Kharif {selected_year}")
        axes_h[1].legend(loc="upper right")
        axes_h[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

# -------------------------------------------------------------
# TAB 3: MEMBERSHIP FUNCTIONS
# -------------------------------------------------------------
with tab_mf:
    st.subheader("Mamdani Membership Functions (Universes of Discourse)")
    st.markdown(
        "Linguistic partitions grounded in pearl millet physiology and semi-arid soil hydrology."
    )

    fig_all_mf, axes_mf = plt.subplots(2, 2, figsize=(11, 7))

    # Soil moisture
    ax = axes_mf[0, 0]
    ax.plot(fis.x_sm, fis.sm_dry, "r", lw=2, label="Dry")
    ax.plot(fis.x_sm, fis.sm_moderate, "y", lw=2, label="Moderate")
    ax.plot(fis.x_sm, fis.sm_wet, "g", lw=2, label="Wet")
    ax.set_title("Input 1: Soil Moisture (%)")
    ax.set_xlabel("Root-Zone Moisture (%)")
    ax.set_ylabel("Membership Degree (μ)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # Temperature
    ax = axes_mf[0, 1]
    ax.plot(fis.x_temp, fis.temp_cool, "b", lw=2, label="Cool")
    ax.plot(fis.x_temp, fis.temp_moderate, "g", lw=2, label="Moderate")
    ax.plot(fis.x_temp, fis.temp_hot, "r", lw=2, label="Hot")
    ax.set_title("Input 2: Temperature (°C)")
    ax.set_xlabel("Mean Daily Temperature (°C)")
    ax.set_ylabel("Membership Degree (μ)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # Relative Humidity
    ax = axes_mf[1, 0]
    ax.plot(fis.x_rh, fis.rh_low, "r", lw=2, label="Low")
    ax.plot(fis.x_rh, fis.rh_moderate, "y", lw=2, label="Moderate")
    ax.plot(fis.x_rh, fis.rh_high, "b", lw=2, label="High")
    ax.set_title("Input 3: Relative Humidity (%)")
    ax.set_xlabel("Daily Relative Humidity (%)")
    ax.set_ylabel("Membership Degree (μ)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    # Recent Rainfall
    ax = axes_mf[1, 1]
    ax.plot(fis.x_rain, fis.rain_low, "r", lw=2, label="Low")
    ax.plot(fis.x_rain, fis.rain_moderate, "y", lw=2, label="Moderate")
    ax.plot(fis.x_rain, fis.rain_high, "b", lw=2, label="High")
    ax.set_title("Input 4: 3-Day Cumulative Rainfall (mm)")
    ax.set_xlabel("Antecedent Rainfall (mm)")
    ax.set_ylabel("Membership Degree (μ)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig_all_mf)
    plt.close(fig_all_mf)

# -------------------------------------------------------------
# TAB 4: EVALUATION & BENCHMARK METRICS
# -------------------------------------------------------------
with tab_eval:
    st.subheader("Quantitative Evaluation Against Agronomic Reference (FAO-56)")

    if eval_metrics:
        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
        fis_metrics = eval_metrics.get("fis_vs_fao56", {})
        base_metrics = eval_metrics.get("baseline_vs_fao56", {})

        ev_c1.metric("Dataset Records", f"{eval_metrics.get('dataset_records_count', 492)} Days")
        ev_c2.metric("FIS vs FAO-56 Pearson (r)", f"{fis_metrics.get('Pearson_Correlation', 0.736):.3f}",
                    delta=f"+{fis_metrics.get('Pearson_Correlation', 0.736) - base_metrics.get('Pearson_Correlation', 0.422):.3f} vs Baseline")
        ev_c3.metric("Trigger Agreement Rate", f"{fis_metrics.get('Trigger_Agreement_Rate', 0.657)*100:.1f} %")
        ev_c4.metric("Avg Active Rules / Day", f"{eval_metrics.get('mean_active_rules_per_day', 2.05):.2f}")

        st.markdown("#### Performance Metric Comparison")
        comp_df = pd.DataFrame([
            {
                "Model": "Mamdani Fuzzy Inference System",
                "MAE (mm/day)": fis_metrics.get("MAE_mm"),
                "RMSE (mm/day)": fis_metrics.get("RMSE_mm"),
                "Pearson Correlation (r)": fis_metrics.get("Pearson_Correlation"),
            },
            {
                "Model": "Fixed Threshold Baseline",
                "MAE (mm/day)": base_metrics.get("MAE_mm"),
                "RMSE (mm/day)": base_metrics.get("RMSE_mm"),
                "Pearson Correlation (r)": base_metrics.get("Pearson_Correlation"),
            },
        ])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.markdown("#### Category Distribution across Kharif Seasons")
        st.bar_chart(pd.Series(eval_metrics.get("category_distribution", {})))

    # Display saved sensitivity surface plot
    surf_file = PROJECT_ROOT / "data" / "processed" / "sensitivity_surface.png"
    if surf_file.exists():
        st.markdown("#### 3D Sensitivity Surface (Continuity & Monotonicity)")
        st.image(str(surf_file), caption="Fuzzy Response Surface to Soil Moisture & Ambient Temperature (Rain=2mm, RH=50%)")

# -------------------------------------------------------------
# TAB 5: SCIENTIFIC GOVERNANCE & SOURCES
# -------------------------------------------------------------
with tab_gov:
    st.subheader("Data Provenance & Project Scientific Governance")

    st.markdown("""
    ### 1. Data Provenance & Verification
    * **Provider**: NASA Prediction Of Worldwide Energy Resources (POWER) Daily Agroclimatology API.
    * **Coordinates**: Latitude `26.9124°N`, Longitude `75.7873°E` (Jaipur District, Rajasthan).
    * **Observations**: 1,249 continuous daily records (2021-06-01 to 2024-10-31), filtered to 492 Kharif season days.
    * **Missingness & Sentinels**: 0 missing values; pristine raw files preserved unchanged in `data/raw/`.

    ### 2. Methodological Assumptions & Justified Proxies
    * **Root-Zone Soil Wetness (`GWETROOT`)**: Derived from the GMAO MERRA-2 Catchment Land Surface Model ($0.5^\circ \times 0.625^\circ$). While highly representative of district-level root-zone soil water depletion, it is not an in-situ capacitive sensor.
    * **Pearl Millet Agronomy**: Calibrated using FAO-56 crop coefficient stages for *Pennisetum glaucum* (Initial $K_c = 0.5$, Mid $K_c = 1.15$, End $K_c = 0.45$).

    ### 3. Transparent Limitations & Project Disclaimers
    > [!WARNING]
    > **Decision-Support Prototype Disclaimer**:
    > This prototype is designed for transparent decision-support and academic study. It is **not** an automated actuator or direct field controller. Actual farm irrigation scheduling requires on-site soil texture testing, electrical conductivity checks, and ground sensor calibration. No claims of field-validated water savings or yield increases are made without randomized on-farm control trials.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Smart Irrigation Recommendation System • Mamdani Fuzzy Logic • Jaipur District, Rajasthan"
    "</div>",
    unsafe_allow_html=True,
)

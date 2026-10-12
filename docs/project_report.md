# Project Report: Smart Irrigation Recommendation System for Pearl Millet Cultivation in Jaipur District, Rajasthan

**Domain:** Artificial Intelligence & Soft Computing — Mamdani Fuzzy Inference System  
**Status:** Implementation complete and independently verified  

---

## Abstract

Agricultural irrigation scheduling in water-scarce semi-arid regions requires balancing crop physiological demand, soil moisture depletion, antecedent rainfall, and atmospheric evaporative conditions. Conventional fixed-threshold triggers fail at boundary conditions and cannot represent the inherent multi-dimensional uncertainty of natural agroclimatic variables.

This project implements, evaluates, and documents an interpretable **Mamdani Fuzzy Inference System (FIS)** for daily irrigation recommendation for pearl millet (*Pennisetum glaucum* / *Bajra*) in Jaipur District, Rajasthan, India.

The system is driven entirely by **verified historical data** from the NASA POWER Daily Agroclimatology API (1,249 daily records for Jaipur from 2021 to 2024; 492 Kharif crop season observations across four growing seasons). Four environmental drivers — root-zone soil moisture, ambient temperature, relative humidity, and 3-day antecedent rainfall — are processed through piecewise-linear fuzzy membership functions and 17 agronomic IF–THEN rules. Defuzzification via the centroid method yields a net daily irrigation recommendation (0–12 mm/day), adjusted for crop growth stage.

Evaluation against the standard **FAO-56 Penman-Monteith Crop Evapotranspiration Reference** (Allen et al., 1998) across four real Kharif seasons demonstrates a Pearson correlation of $r = 0.823$ and an 81.5% binary trigger agreement rate — substantially stronger than a fixed-threshold baseline ($r = 0.309$). An interactive Streamlit dashboard exposes membership functions, rule firing strengths, and centroid defuzzification curves in real time.

> **Scientific caveat:** These metrics measure agreement with a meteorological reference equation calculated from the same reanalysis weather data. They do **not** constitute field-validated water savings, yield improvements, or agronomic efficacy claims. No field trials have been conducted.

---

## 1. Introduction and Problem Statement

### 1.1 Regional Context
Jaipur district (26.91°N, 75.79°E, elevation 435 m a.s.l.) falls within the semi-arid agroclimatic zone of eastern Rajasthan. The region experiences erratic Southwest monsoon precipitation (annual average 550–650 mm), peak summer temperatures exceeding 40°C, and predominantly sandy loam soils with low water-holding capacity. Pearl millet (*bajra*) occupies the largest cropped area in the district during the Kharif (June–October) season.

### 1.2 Irrigation Challenge
Sandy loam soils drain rapidly; under-irrigation during the peak vegetative and flowering phases causes disproportionate yield loss (40–60%), while over-irrigation during early germination promotes soil crusting and root disease. Farmers typically lack quantified environmental monitoring and rely on visual plant-stress cues or fixed calendar schedules — both suboptimal.

### 1.3 Motivation for Fuzzy Logic
Crisp threshold rules produce abrupt, context-insensitive decisions. A temperature of 35.1°C triggers the same response as 37°C, ignoring whether soil is already saturated from recent rainfall or whether humidity is suppressing transpiration. Fuzzy logic:
* Permits smooth, graduated recommendations that respond proportionally to each environmental driver.
* Produces interpretable, auditable rules that agronomists can inspect and validate.
* Avoids the training-data requirements and black-box opacity of machine learning regressors.

---

## 2. Data Acquisition and Preprocessing

### 2.1 Data Source
All observations were downloaded directly from the **NASA POWER Daily Agroclimatology API** (v2.10.0).

| Parameter | Code | Unit | Physical Layer |
|---|---|---|---|
| Root-Zone Soil Wetness | `GWETROOT` | fraction 0–1 | MERRA-2 Catchment LSM |
| Mean 2m Temperature | `T2M` | °C | MERRA-2 reanalysis |
| Max 2m Temperature | `T2M_MAX` | °C | MERRA-2 reanalysis |
| Min 2m Temperature | `T2M_MIN` | °C | MERRA-2 reanalysis |
| Relative Humidity | `RH2M` | % | MERRA-2 reanalysis |
| Wind Speed | `WS2M` | m/s | MERRA-2 reanalysis |
| Solar Radiation | `ALLSKY_SFC_SW_DWN` | MJ/m²/day | CERES / GEOS-FP |
| Precipitation | `PRECTOTCORR` | mm/day | GPCP / MERRA-2 |

* **Period:** 2021-06-01 to 2024-10-31 — 1,249 continuous daily records.
* **Kharif Subset:** 492 days (June 15 – October 15 across four seasons).
* **Missing data:** 0 missing records and 0 fill-value sentinels (−999.0) detected.
* **Raw files preserved unmodified** in `data/raw/`.

> **Soil moisture note:** `GWETROOT` represents model-simulated root-zone wetness at 0.5° × 0.625° spatial resolution — not an in-situ capacitance probe. It is used as a relative indicator of root-zone soil water availability.

### 2.2 Preprocessing Steps
1. ISO date parsing and duplicate-date removal.
2. Sentinel value replacement (−999.0 → NaN) and forward-fill interpolation.
3. Physical bounds verification (0 anomalies found).
4. Moisture scaling: `GWETROOT` × 100 → `soil_moisture_pct` (0–100%).
5. 3-day antecedent rainfall sum: `rainfall_3d_sum`.
6. Phenological crop stage proxy assignment (`Vegetative` / `Flowering` / `Maturity`).

---

## 3. Mamdani Fuzzy Inference System

### 3.1 Membership Functions
Four input variables and one output, each partitioned into three or four fuzzy linguistic sets using trapezoidal and triangular membership functions. Full parameter tables are documented in [methodology.md](methodology.md).

### 3.2 Rule Base
17 agronomic IF–THEN rules grounded in FAO-56 crop water requirements and semi-arid sandy loam soil hydrology. Rules are evaluated using the minimum T-norm (AND) and maximum S-norm (OR), with Mamdani min-clipping implication and max-aggregation.

### 3.3 Defuzzification
Centroid (Centre of Gravity) method across the 601-point universe $[0, 12]$ mm/day. Final recommendation adjusted by a growth-stage multiplier (Vegetative: 0.85; Flowering: 1.00; Maturity: 0.65).

---

## 4. Reference Evaluation Methodology

### 4.1 FAO-56 Penman-Monteith Reference ET0
Implemented strictly following Allen et al. (1998, Chapter 3):

$$ET_0 = \frac{0.408\,\Delta\,(R_n - G) + \gamma\,\dfrac{900}{T+273}\,u_2\,(e_s - e_a)}{\Delta + \gamma\,(1 + 0.34\,u_2)}$$

All sub-terms (extraterrestrial radiation, clear-sky radiation, net longwave, saturation and actual vapour pressure, psychrometric constant) are computed from the downloaded NASA POWER variables. Jaipur elevation (435 m a.s.l.) is used for atmospheric pressure.

### 4.2 Crop Water Requirement
$ET_c = K_c \times ET_0$, using FAO-56 Table 12 crop coefficients: $K_c = 0.55$ (Vegetative), $K_c = 1.15$ (Flowering), $K_c = 0.45$ (Maturity).

Effective rainfall applied using the simplified USDA SCS daily method ($P_\text{eff} = 0.80\,P$ for $P \geq 5$ mm; 0 otherwise). Net Irrigation Requirement $= \max(0,\ ET_c - P_\text{eff})$, with a soil-saturation override at soil moisture $\geq 65$%.

> **Note:** An earlier draft used a Hargreaves-Samani temperature-radiation approximation. The current implementation uses the complete Penman-Monteith equation, which is the internationally recognised standard for FAO-56 reference ET.

---

## 5. Results

### 5.1 Verified Evaluation Metrics (Jaipur Kharif 2021–2024)

| Metric | Mamdani FIS | Fixed Threshold Baseline |
|---|:---:|:---:|
| **Pearson Correlation $r$** | **0.823** | 0.309 |
| MAE (mm/day) | 1.217 | 1.478 |
| RMSE (mm/day) | 1.537 | 2.465 |
| Trigger Agreement Rate | **81.5%** | — |
| Mean active rules/day | 2.05 | — |

### 5.2 Recommendation Distribution

Across 492 Kharif days:
* **None** (< 1 mm): 235 days (47.8%) — primarily during monsoon wet spells.
* **Low** (1–4 mm): 117 days (23.8%) — mild depletion, vegetative stages.
* **Medium** (4–7.5 mm): 135 days (27.4%) — peak flowering consumption.
* **High** (> 7.5 mm): 5 days (1.0%) — severe dry-spell drought events.

---

## 6. Software Architecture

```
smart-irrigation-fuzzy/
├── data/raw/           — Immutable NASA POWER downloads (JSON + CSV)
├── data/processed/     — Cleaned datasets, metrics, evaluation plots
├── docs/               — Methodology, project report, viva notes, data sources
├── src/
│   ├── data_loader.py  — NASA POWER API fetcher
│   ├── preprocessing.py — Cleaning, validation, feature engineering
│   ├── fuzzy_system.py — Mamdani FIS (scikit-fuzzy + custom explainability)
│   └── evaluation.py   — FAO-56 PM reference, baseline, metrics, sensitivity
├── app/streamlit_app.py — 5-tab interactive Streamlit dashboard
├── tests/
│   ├── test_fuzzy_system.py   — 10 logic and boundary tests
│   ├── test_preprocessing.py  — 5 pipeline tests
│   └── test_evaluation.py     — 5 reference model tests
└── requirements.txt
```

**Test suite:** 20 automated tests — **all pass**.

---

## 7. Limitations and Future Work

1. **Soil moisture proxy:** `GWETROOT` is a 0.5° × 0.625° reanalysis model output, not a point in-situ sensor. Field deployment would require local sensor calibration.
2. **Advisory prototype:** Not an automated irrigation actuator.
3. **No field trials:** Evaluation is against a meteorological reference equation. No water saving or yield claims are made.
4. **Spatial resolution:** Grid-cell data may not represent micro-topographic or soil textural variation within Jaipur district.
5. **Future enhancements:** Real-time NDVI integration for dynamic $K_c$ adjustment; fuzzy rule optimisation using historical yield data; ANFIS-based refinement.

---

## 8. References

* Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements.* FAO Irrigation and Drainage Paper No. 56. Food and Agriculture Organization, Rome.
* Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
* NASA POWER Project. (2024). *POWER Daily Data*. NASA Langley Research Center. https://power.larc.nasa.gov/
* ICAR. Pearl Millet (*Bajra*) crop production guide. Indian Council of Agricultural Research.

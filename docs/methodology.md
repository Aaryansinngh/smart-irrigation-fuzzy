# Methodology: Mamdani Fuzzy Inference System for Pearl Millet Irrigation

**Project Title:** Smart Irrigation Recommendation System for Pearl Millet (*Pennisetum glaucum*) Cultivation in Jaipur District, Rajasthan  
**Core Soft Computing Technique:** Mamdani Fuzzy Inference System (FIS)  
**Defuzzification:** Centroid Method (Centre of Gravity, COG)  
**Reference ET Model:** FAO-56 Penman-Monteith (Allen et al., 1998, Chapter 3)

---

## 1. Agronomic Rationale & Regional Context

### 1.1 Crop Characteristics: Pearl Millet (*Bajra*)
Pearl millet is the primary staple Kharif crop in the semi-arid zone of Rajasthan.

* **Crop duration:** 75–90 days.
* **Sowing window:** Mid-June through mid-July, aligned with Southwest monsoon onset.
* **Soil type in Jaipur district:** Predominantly sandy loam with high infiltration rates, low water-holding capacity, and low organic matter.
* **Total seasonal water requirement:** 350–500 mm.
* **Critical moisture windows:** Tillering (20–25 days after sowing) and especially **booting / flowering / grain filling** (40–60 DAS). Severe water stress during flowering can reduce yield by 40–60% (ICAR / FAO-56 guidance). Over-watering during early germination on sandy soils promotes root disease.

### 1.2 Motivation for Fuzzy Logic over Crisp Thresholds
Traditional irrigation scheduling relies on binary triggers (e.g., *irrigate when soil moisture < 35%*). Such crisp models fail at boundary conditions:
* A reading of 34.9% triggers immediate irrigation while 35.1% triggers none, regardless of ambient temperature, humidity, or antecedent rain.
* Fuzzy logic provides smooth, continuous transitions between linguistic states and naturally synthesises multi-dimensional environmental drivers without the data hunger or opacity of black-box neural networks.
* Mamdani FIS rules remain legible to agronomists and extension workers, supporting auditability and trust.

---

## 2. Data Acquisition and Feature Engineering

### 2.1 Source
All input variables are derived from the **NASA POWER Daily Agroclimatology API** (v2.10.0), NASA Langley Research Center. No observation was fabricated.

| Variable | NASA Parameter | Unit | Model Layer |
|---|---|---|---|
| Root-Zone Soil Moisture | `GWETROOT` | fraction (0–1) | MERRA-2 Catchment Land Surface Model |
| Daily Mean Temperature | `T2M` | °C | MERRA-2 reanalysis |
| Maximum Temperature | `T2M_MAX` | °C | MERRA-2 reanalysis |
| Minimum Temperature | `T2M_MIN` | °C | MERRA-2 reanalysis |
| Relative Humidity | `RH2M` | % | MERRA-2 reanalysis |
| Wind Speed | `WS2M` | m/s | MERRA-2 reanalysis |
| Solar Radiation | `ALLSKY_SFC_SW_DWN` | MJ/m²/day | CERES / GEOS-FP |
| Precipitation | `PRECTOTCORR` | mm/day | GPCP / MERRA-2 |

> **Soil moisture proxy note:** `GWETROOT` is a model-derived normalized root-zone wetness index from the MERRA-2 Catchment Land Surface Model (spatial resolution 0.5° × 0.625°). It correlates closely with available soil-water depletion but is **not** an in-situ capacitance probe measurement. It is used in this prototype as a relative moisture indicator.

### 2.2 Preprocessing Pipeline
1. **Schema & Fill-Value Validation** — detects and replaces NASA's −999.0 sentinel fill code.
2. **Physical Bounds Verification** — 0 anomalies detected across the 2021–2024 Jaipur dataset.
3. **Moisture Scaling** — `GWETROOT` (0.0–1.0) scaled to 0–100% soil moisture percentage.
4. **Antecedent Rainfall Window** — 3-day cumulative precipitation rolling sum (`rainfall_3d_sum`).
5. **Phenological Staging** — calendar-based crop growth stage proxy:
   * Vegetative: 15 Jun – 25 Jul
   * Flowering / Grain Filling: 26 Jul – 15 Sep (water-critical period)
   * Maturity: 16 Sep – 15 Oct

---

## 3. Mamdani Fuzzy Inference System Design

### 3.1 Input Variables & Membership Functions

All membership functions use standard piecewise-linear forms (trapezoidal, triangular):

#### Input 1: Root-Zone Soil Moisture ($SM$), Universe $[0, 100]\%$

| Linguistic Term | MF Type | Parameters |
|---|---|---|
| `Dry` | trapmf | $[0, 0, 20, 38]$ |
| `Moderate` | trimf | $[28, 48, 68]$ |
| `Wet` | trapmf | $[58, 75, 100, 100]$ |

#### Input 2: Daily Mean Temperature ($T$), Universe $[15, 48]$°C

| Linguistic Term | MF Type | Parameters |
|---|---|---|
| `Cool` | trapmf | $[15, 15, 22, 28]$ |
| `Moderate` | trimf | $[25, 32, 38]$ |
| `Hot` | trapmf | $[35, 40, 48, 48]$ |

#### Input 3: Relative Humidity ($RH$), Universe $[10, 100]\%$

| Linguistic Term | MF Type | Parameters |
|---|---|---|
| `Low` | trapmf | $[10, 10, 25, 45]$ |
| `Moderate` | trimf | $[35, 55, 75]$ |
| `High` | trapmf | $[65, 80, 100, 100]$ |

#### Input 4: 3-Day Cumulative Rainfall ($R$), Universe $[0, 60]$ mm

| Linguistic Term | MF Type | Parameters |
|---|---|---|
| `Low` | trapmf | $[0, 0, 5, 12]$ |
| `Moderate` | trimf | $[8, 20, 35]$ |
| `High` | trapmf | $[28, 45, 60, 60]$ |

#### Output: Daily Net Irrigation Requirement ($I$), Universe $[0, 12]$ mm/day

| Linguistic Term | MF Type | Parameters |
|---|---|---|
| `None` | trapmf | $[0, 0, 0.5, 2.0]$ |
| `Low` | trimf | $[1.0, 3.0, 5.0]$ |
| `Medium` | trimf | $[4.0, 6.5, 9.0]$ |
| `High` | trapmf | $[7.5, 9.5, 12.0, 12.0]$ |

---

### 3.2 Agronomic Rule Base (17 IF–THEN Rules)

| Rule | Antecedent | Consequent | Agronomic Rationale |
|:---:|---|:---:|---|
| R1 | Soil is **Wet** OR Rain is **High** | **None** | Saturation / heavy rain fully satisfies demand; additional irrigation risks leaching on sandy loam. |
| R2 | Soil is **Wet** AND Rain is **Moderate** | **None** | Adequate soil buffer replenished. |
| R3 | Soil is **Wet** AND Rain is **Low** AND Temp is **Cool** | **None** | Low evaporative demand; soil reserves sufficient. |
| R4 | Soil is **Wet** AND Rain is **Low** AND Temp is **Hot** | **Low** | Hot weather begins surface desiccation; light preventive top-up. |
| R5 | Soil is **Moderate** AND Rain is **High** | **None** | Recent rain recharges root zone to field capacity. |
| R6 | Soil is **Moderate** AND Rain is **Moderate** AND RH is **High** | **None** | Vapour pressure deficit low; transpiration suppressed. |
| R7 | Soil is **Moderate** AND Rain is **Moderate** AND RH is **Moderate** | **Low** | Minor supplemental top-up for sustained vegetative growth. |
| R8 | Soil is **Moderate** AND Rain is **Low** AND Temp is **Cool** | **Low** | Mild demand; light schedule sufficient. |
| R9 | Soil is **Moderate** AND Rain is **Low** AND Temp is **Moderate** | **Medium** | Standard mid-season evapotranspiration demand. |
| R10 | Soil is **Moderate** AND Rain is **Low** AND Temp is **Hot** AND RH is **Low** | **High** | Maximum vapour pressure deficit; rapid soil desiccation. |
| R11 | Soil is **Moderate** AND Rain is **Low** AND Temp is **Hot** AND RH is **Moderate** | **Medium** | High demand buffered by moderate humidity. |
| R12 | Soil is **Dry** AND Rain is **High** | **Low** | Rain wets topsoil but dry subsoil still requires light follow-up. |
| R13 | Soil is **Dry** AND Rain is **Moderate** | **Medium** | Partial recharge; supplemental irrigation required. |
| R14 | Soil is **Dry** AND Rain is **Low** AND Temp is **Cool** | **Medium** | Root zone depleted; re-wetting needed even in cooler weather. |
| R15 | Soil is **Dry** AND Rain is **Low** AND Temp is **Moderate** | **High** | Critical deficit during active growth. |
| R16 | Soil is **Dry** AND Rain is **Low** AND Temp is **Hot** | **High** | Emergency drought condition; immediate replenishment required. |
| R17 | Soil is **Dry** AND RH is **Low** | **High** | Compound arid condition; high transpiration rate. |

---

## 4. Inference and Defuzzification

### 4.1 Fuzzy Operators
* **Fuzzy AND (T-norm):** Minimum — $\mu_{A \cap B} = \min(\mu_A, \mu_B)$
* **Fuzzy OR (S-norm):** Maximum — $\mu_{A \cup B} = \max(\mu_A, \mu_B)$

### 4.2 Mamdani Implication (Min-clipping)
For each rule $k$ with premise firing strength $\alpha_k$:
$$\mu_{B'_k}(y) = \min\!\left(\alpha_k,\; \mu_{B_k}(y)\right)$$

### 4.3 Max-Aggregation
$$\mu_\text{agg}(y) = \max_{k=1}^{17} \mu_{B'_k}(y)$$

### 4.4 Centroid Defuzzification (Centre of Gravity)
$$I^* = \frac{\displaystyle\int_0^{12} y \cdot \mu_\text{agg}(y)\,dy}{\displaystyle\int_0^{12} \mu_\text{agg}(y)\,dy} \approx \frac{\displaystyle\sum_i y_i \cdot \mu_\text{agg}(y_i)}{\displaystyle\sum_i \mu_\text{agg}(y_i)}$$

### 4.5 Phenological Stage Adjustment
$$I_\text{final} = I^* \times K_\text{stage}$$

| Stage | $K_\text{stage}$ | Rationale |
|---|---|---|
| Vegetative | 0.85 | Moderate demand; crop not yet at peak transpiration. |
| Flowering | 1.00 | Reference calibration point; highest crop coefficient. |
| Maturity | 0.65 | Demand declining; grain desiccation begins. |
| Off-Season | 0.00 | No irrigation applied outside Kharif window. |

---

## 5. Reference Evapotranspiration Model

### 5.1 FAO-56 Penman-Monteith Equation (Allen et al., 1998)

$$ET_0 = \frac{0.408\,\Delta\,(R_n - G) + \gamma\,\dfrac{900}{T + 273}\,u_2\,(e_s - e_a)}{\Delta + \gamma\,(1 + 0.34\,u_2)}$$

Where:
* $\Delta$ — slope of saturation vapour pressure curve (kPa/°C)
* $R_n$ — net radiation at crop surface (MJ/m²/day) — computed from incoming solar radiation, clear-sky radiation estimate, and longwave emission terms
* $G$ — soil heat flux (MJ/m²/day) — assumed $\approx 0$ for daily time steps
* $\gamma$ — psychrometric constant (kPa/°C), computed from site elevation (Jaipur: 435 m a.s.l.)
* $u_2$ — wind speed at 2 m height (m/s)
* $e_s$ — saturation vapour pressure (kPa)
* $e_a$ — actual vapour pressure (kPa), derived from relative humidity

Crop evapotranspiration: $ET_c = K_c \times ET_0$ using stage-specific $K_c$ values from FAO-56 Table 12.

**This is the standard FAO-56 Penman-Monteith equation.** The previous version of `evaluation.py` used a Hargreaves-Samani approximation. That has been corrected in the current implementation.

---

## 6. Evaluation Methodology & Verified Results

The FIS was evaluated against the FAO-56 Penman-Monteith ETc reference model across **492 verified Kharif season observations** (Jaipur, 2021–2024).

| Metric | Mamdani Fuzzy System | Fixed Threshold Baseline |
|---|:---:|:---:|
| Pearson Correlation $r$ with FAO-56 ETc | **0.823** | 0.309 |
| MAE (mm/day) | 1.217 | 1.478 |
| RMSE (mm/day) | 1.537 | 2.465 |
| Irrigation Trigger Agreement | **81.5%** | — |
| Mean concurrent active rules/day | 2.05 | — |

> **Interpretation note:** These metrics measure the agreement between the FIS's rule-based recommendation and a theoretical crop evapotranspiration reference calculated from reanalysis weather data. They do **not** measure field-measured water savings, yield improvements, or validated sensor accuracy. No field trials have been conducted.

---

## 7. Scientific Assumptions & Limitations

1. **Soil moisture proxy:** `GWETROOT` is a reanalysis model output, not an in-situ sensor.
2. **Advisory prototype only:** Not an automated valve controller or yield-prediction tool.
3. **Agronomic Kc values:** Sourced from FAO-56 Table 12 (generic semi-arid pearl millet); local cultivar adjustments have not been applied.
4. **No field validation:** All evaluation is against a meteorological reference equation, not against measured crop water uptake or yield data.
5. **Spatial resolution:** NASA POWER data represents a 0.5° × 0.625° grid cell centred on Jaipur, not a specific farm field.

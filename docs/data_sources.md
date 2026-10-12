# Data Source Register & Verification Notes

**Project:** Smart Irrigation Recommendation System for Pearl Millet Cultivation in Jaipur District, Rajasthan  
**Core Method:** Mamdani Fuzzy Inference System (FIS)  
**Verification Date:** 12 October 2026  

---

## 1. Candidate Source A: NASA POWER Agroclimatology (Verified & Preserved)

* **Source Name:** NASA Prediction Of Worldwide Energy Resources (POWER) Daily Agroclimatology API
* **API Documentation:** https://power.larc.nasa.gov/docs/services/api/temporal/daily/
* **API Endpoint:** `https://power.larc.nasa.gov/api/temporal/daily/point`
* **Target Coordinates:**
  * Latitude: `26.9124° N`
  * Longitude: `75.7873° E`
  * Geographic Area: Jaipur District, Rajasthan, India
* **Data Access Protocol:** RESTful HTTP GET request, JSON & tabular CSV formats.
* **Date Range Acquired:** 2021-06-01 to 2024-10-31 (1,249 daily records, encompassing 4 complete Kharif cropping cycles: 2021, 2022, 2023, and 2024).
* **Data Provider & Models:** NASA Langley Research Center / GMAO MERRA-2 and GEOS-FP land data assimilation systems.

### Acquired Variables and Parameter Register

| Variable Code | Parameter Description | Units | Source Grid / Sensor | Agronomic Purpose in System |
|---|---|---|---|---|
| `T2M` | Daily Mean Temperature at 2m | °C | MERRA-2 ($0.5^\circ \times 0.625^\circ$) | Input to Fuzzy Inference & Evaporative Demand |
| `T2M_MAX` | Daily Maximum Temperature at 2m | °C | MERRA-2 ($0.5^\circ \times 0.625^\circ$) | Heat stress analysis & FAO-56 Reference ET |
| `T2M_MIN` | Daily Minimum Temperature at 2m | °C | MERRA-2 ($0.5^\circ \times 0.625^\circ$) | Diurnal range & FAO-56 Reference ET |
| `RH2M` | Relative Humidity at 2m | % | MERRA-2 ($0.5^\circ \times 0.625^\circ$) | Atmospheric moisture demand / Vapour pressure |
| `PRECTOTCORR`| Corrected Daily Precipitation | mm/day | GPCP / MERRA-2 | Antecedent rainfall & 3-day moisture window |
| `GWETTOP` | Surface Soil Wetness (0–5 cm) | fraction (0–1) | MERRA-2 Land Model | Topsoil moisture status / Seedbed germination |
| `GWETROOT` | Root Zone Soil Wetness (0–100 cm) | fraction (0–1) | MERRA-2 Land Model | Primary soil moisture input for pearl millet root zone |
| `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Solar Insolation | $\text{MJ/m}^2/\text{day}$ | CERES / GEOS-FP | Solar radiation component for FAO-56 Penman-Monteith |
| `WS2M` | Wind Speed at 2m | m/s | MERRA-2 | Aerodynamic term for FAO-56 Penman-Monteith |

### Verification Status & Quality Inspection
* **Missingness:** 0 missing dates in the acquired period (1,249 continuous daily records).
* **Fill Values:** API default missing code is `-999.0`. No `-999.0` flags were encountered across the queried parameters.
* **Pristine Preservation:**
  * Raw JSON: `data/raw/nasa_power_jaipur_raw.json`
  * Raw CSV: `data/raw/nasa_power_jaipur_raw.csv`
* **Caveat & Soil Moisture Context:** NASA POWER `GWETROOT` and `GWETTOP` represent normalized wetness indices ranging from 0.0 (completely dry, below wilting point) to 1.0 (saturation) produced by the Catchment Land Surface Model. While this correlates strongly with root-zone available water capacity, it is a model-derived reanalysis product rather than a point in-situ capacitance probe. In the system, it is treated as root-zone relative moisture availability.

---

## 2. Candidate Source B: Government of India / NRSC Bhuvan Portals (Catalog Review)

* **Catalog 1:** Government of India Open Government Data (OGD) Soil Moisture Catalog (`https://ap.data.gov.in/catalog/soil-moisture` and `data.gov.in`)
* **Catalog 2:** NRSC / ISRO Bhuvan National Hydrology Project (`https://bhuvan.nrsc.gov.in/nhp/nhp-data-download`)
* **Catalog 3:** MOSDAC ISRO SMAP Soil Wetness Index (`https://mosdac.gov.in`)

### Inspection & Feasibility Findings
1. **Access Constraints:**
   * India Open Government Data (OGD) primarily offers periodic state-level bulletins and point agricultural census tables rather than continuous, automated programmatic daily time-series APIs for Jaipur tehsil stations.
   * NRSC Bhuvan National Hydrology Project VIC (Variable Infiltration Capacity) model outputs require registered institutional credentials and manual GeoTIFF tile downloads with raster extraction rather than direct single-station programmatic REST queries.
2. **Resolution Disparity:**
   * Satellite soil wetness from ISRO/MOSDAC (SMAP L4/RISAT) provides regional grids ($0.25^\circ \times 0.25^\circ$), aligned closely with the spatial resolution of NASA POWER's MERRA-2 ($0.5^\circ \times 0.625^\circ$).
3. **Scientific Integrity Decision:**
   * In strict adherence to the project specification ("*If the soil-moisture dataset cannot be verified, do not fabricate it: either find a documented alternative or revise the scope and clearly disclose the limitation*"):
   * We use the verified, peer-reviewed, and programmatically validated NASA POWER Daily Agroclimatology dataset for both atmospheric drivers and root-zone soil wetness.
   * The limitation of satellite/reanalysis-modeled soil wetness vs direct ground probes is transparently declared in system documentation and the Streamlit UI.

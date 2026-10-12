"""
Data Acquisition Module for Smart Irrigation Recommendation System.

Retrieves documented historical meteorological and land-surface soil moisture
observations for Jaipur, Rajasthan from the NASA POWER Daily API.
Saves raw outputs to data/raw/ without modification for auditability.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Geographic & Temporal parameters for Jaipur District, Rajasthan
JAIPUR_LAT = 26.9124
JAIPUR_LON = 75.7873
DEFAULT_START_DATE = "20210601"  # June 1, 2021 (Start of Kharif 2021)
DEFAULT_END_DATE = "20241031"    # Oct 31, 2024 (End of Kharif 2024)

# NASA POWER Daily API Parameters
# T2M: Temperature at 2m (°C)
# T2M_MAX: Max Temperature at 2m (°C)
# T2M_MIN: Min Temperature at 2m (°C)
# RH2M: Relative Humidity at 2m (%)
# PRECTOTCORR: Precipitation Corrected (mm/day)
# GWETTOP: Surface Soil Wetness (0-1 fraction)
# GWETROOT: Root Zone Soil Wetness (0-1 fraction)
# ALLSKY_SFC_SW_DWN: All Sky Insolation Incident on a Horizontal Surface (MJ/m^2/day)
# WS2M: Wind Speed at 2m (m/s)
QUERY_PARAMETERS = [
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

NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power_data(
    latitude: float = JAIPUR_LAT,
    longitude: float = JAIPUR_LON,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    parameters: list[str] = QUERY_PARAMETERS,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Fetch daily time-series data from NASA POWER API.
    
    Returns raw JSON response parsed into a Python dictionary.
    """
    params_str = ",".join(parameters)
    url = (
        f"{NASA_POWER_ENDPOINT}?"
        f"parameters={params_str}&"
        f"community=AG&"
        f"longitude={longitude}&"
        f"latitude={latitude}&"
        f"start={start_date}&"
        f"end={end_date}&"
        f"format=JSON"
    )

    logger.info(f"Requesting NASA POWER API: lat={latitude}, lon={longitude}, range={start_date}-{end_date}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    logger.info("Successfully received payload from NASA POWER.")
    return payload


def save_raw_data(
    payload: Dict[str, Any],
    raw_dir: Path,
    file_prefix: str = "nasa_power_jaipur",
) -> tuple[Path, Path]:
    """Save immutable raw JSON and raw CSV representations of the API response."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"{file_prefix}_raw.json"
    csv_path = raw_dir / f"{file_prefix}_raw.csv"

    # Add download metadata wrapper
    retrieval_metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_endpoint": NASA_POWER_ENDPOINT,
        "payload": payload,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_metadata, f, indent=2)
    logger.info(f"Raw JSON preserved at: {json_path}")

    # Also convert parameter dictionary to raw tabular CSV
    param_dict = payload.get("properties", {}).get("parameter", {})
    df = pd.DataFrame(param_dict)
    df.index.name = "date"
    df.to_csv(csv_path)
    logger.info(f"Raw CSV tabular data preserved at: {csv_path} (records={len(df)})")

    return json_path, csv_path


def load_raw_dataset(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load existing raw CSV if available, or fetch from API if not yet downloaded."""
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    csv_path = raw_dir / "nasa_power_jaipur_raw.csv"
    if csv_path.exists():
        logger.info(f"Loading cached raw CSV from {csv_path}")
        return pd.read_csv(csv_path, index_col="date")

    logger.info("Raw data not found locally. Initiating download...")
    payload = fetch_nasa_power_data()
    _, csv_path = save_raw_data(payload, raw_dir)
    return pd.read_csv(csv_path, index_col="date")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_directory = base_dir / "data" / "raw"
    logger.info("Executing NASA POWER Data Acquisition Pipeline...")
    data_payload = fetch_nasa_power_data()
    save_raw_data(data_payload, raw_directory)
    logger.info("Data acquisition complete.")

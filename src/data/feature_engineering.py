"""
Feature engineering module for meteorological and regime-aware machine learning.
Creates temporal, thermodynamic, dynamic, spatial, and NWP interaction features.
"""

import math
from typing import Dict, List, Any
import numpy as np

FEATURE_NAMES = [
    # NWP Core
    "rainfall_nwp",
    "log_rainfall_nwp",
    # Thermodynamic & Dynamic
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_u",
    "wind_v",
    "moisture_flux",
    "cape",
    "vertical_velocity",
    "dewpoint_approx",
    # Spatial
    "lat",
    "lon",
    "elevation",
    "coast_dist_km",
    "orographic_enhancement_index",
    # Temporal
    "month",
    "day_of_year",
    "is_monsoon_core", # July-August indicator
    "lead_time_hours"
]

def engineer_features_single(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes all engineered meteorological features for a single record.
    ZERO LEAKAGE: No current or future observation-derived features are used.
    """
    rain = max(0.0, float(row.get("rainfall_nwp", 0.0)))
    temp = float(row.get("temperature", 28.0))
    rh = max(0.0, min(100.0, float(row.get("humidity", 75.0))))
    pres = float(row.get("pressure", 1004.0))
    wind = max(0.0, float(row.get("wind_speed", 7.0)))
    wind_dir = float(row.get("wind_direction", 240.0)) # Southwest monsoon winds default ~240°
    cape = max(0.0, float(row.get("cape", 1200.0)))
    omega = float(row.get("vertical_velocity", -0.2))
    elev = max(0.0, float(row.get("elevation", 100.0)))
    coast = max(0.0, float(row.get("coast_dist_km", 150.0)))
    lat = float(row.get("lat", 20.0))
    lon = float(row.get("lon", 80.0))
    month = int(row.get("month", 7))
    day = int(row.get("day", 15))
    day_of_year = int(row.get("day_of_year", 196))
    lead_time = float(row.get("lead_time_hours", 24.0))
    
    # Wind components: U (zonal) and V (meridional)
    rad = math.radians(wind_dir)
    wind_u = -wind * math.sin(rad)
    wind_v = -wind * math.cos(rad)
    
    # Moisture flux proxy: wind * specific humidity proxy
    moisture_flux = wind * (rh / 100.0)
    
    # Lawrence (2005) simple dew point approximation
    dewpoint = temp - ((100.0 - rh) / 5.0)
    
    # Orographic enhancement index: onshore/westerly wind hitting Western Ghats or Himalayas
    # Positive westerly wind component (wind_u > 0) perpendicular to Western Ghats
    onshore_westerly = max(0.0, wind_u)
    orographic_enhancement = (elev / 500.0) * (onshore_westerly / 5.0) * (rh / 80.0)
    
    is_monsoon_core = 1.0 if month in [7, 8] else 0.0
    
    return {
        "rainfall_nwp": rain,
        "log_rainfall_nwp": math.log1p(rain),
        "temperature": temp,
        "humidity": rh,
        "pressure": pres,
        "wind_speed": wind,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "moisture_flux": moisture_flux,
        "cape": cape,
        "vertical_velocity": omega,
        "dewpoint_approx": dewpoint,
        "lat": lat,
        "lon": lon,
        "elevation": elev,
        "coast_dist_km": coast,
        "orographic_enhancement_index": orographic_enhancement,
        "month": float(month),
        "day_of_year": float(day_of_year),
        "is_monsoon_core": is_monsoon_core,
        "lead_time_hours": lead_time
    }

def engineer_features_dataset(records: List[Dict[str, Any]]) -> np.ndarray:
    """
    Transforms a list of raw records into a 2D NumPy feature array.
    """
    matrix = []
    for r in records:
        feat_dict = engineer_features_single(r)
        vector = [feat_dict[k] for k in FEATURE_NAMES]
        matrix.append(vector)
    return np.array(matrix, dtype=np.float32)

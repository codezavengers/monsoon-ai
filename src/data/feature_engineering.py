"""
Feature Engineering for Indian Monsoon Numerical Weather Prediction (NWP).
Derives 21 meteorological, kinematic, thermodynamic, and geospatial predictors.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "rainfall_nwp",
    "log_rainfall_nwp",
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
    "lat",
    "lon",
    "elevation",
    "coast_dist_km",
    "orographic_enhancement_index",
    "month",
    "day_of_year",
    "is_monsoon_core",
    "lead_time_hours"
]

def engineer_features_dataset(df_or_record: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Computes all standard features on a DataFrame or single record dictionary.
    """
    if isinstance(df_or_record, dict):
        rec = dict(df_or_record)
        rain = max(0.0, float(rec.get("rainfall_nwp", rec.get("rainfall", 0.0))))
        rec["rainfall_nwp"] = rain
        rec["log_rainfall_nwp"] = float(np.log1p(rain))
        
        rh = min(100.0, max(0.0, float(rec.get("humidity", 75.0))))
        rec["humidity"] = rh
        
        temp = float(rec.get("temperature", 26.0))
        rec["temperature"] = temp
        rec["dewpoint_approx"] = float(temp - ((100.0 - rh) / 5.0))
        
        u = float(rec.get("wind_u", 0.0))
        v = float(rec.get("wind_v", 0.0))
        wind_spd = float(rec.get("wind_speed", 0.0))
        if wind_spd <= 0.0 and (u != 0.0 or v != 0.0):
            wind_spd = float(np.sqrt(u**2 + v**2))
        elif wind_spd > 0.0 and u == 0.0 and v == 0.0:
            u = float(wind_spd * 0.8) # Representative southwest monsoon u-wind
            v = float(wind_spd * 0.6) # Representative southwest monsoon v-wind
        rec["wind_speed"] = max(0.0, wind_spd)
        rec["wind_u"] = u
        rec["wind_v"] = v
        rec["moisture_flux"] = float(rec["wind_speed"] * (rh / 100.0))
        
        elev = max(0.0, float(rec.get("elevation", 0.0)))
        rec["elevation"] = elev
        omega = float(rec.get("vertical_velocity", -0.2))
        rec["vertical_velocity"] = omega
        rec["orographic_enhancement_index"] = float((elev / 500.0) * max(0.0, -omega * 10.0))
        
        lat = float(rec.get("lat", rec.get("latitude", 20.0)))
        lon = float(rec.get("lon", rec.get("longitude", 78.0)))
        rec["lat"] = lat
        rec["lon"] = lon
        rec["is_monsoon_core"] = 1.0 if (18.0 <= lat <= 26.0 and 74.0 <= lon <= 86.0) else 0.0
        
        rec["pressure"] = float(rec.get("pressure", 1000.0))
        rec["cape"] = max(0.0, float(rec.get("cape", 1500.0)))
        rec["coast_dist_km"] = max(0.0, float(rec.get("coast_dist_km", 100.0)))
        rec["month"] = float(rec.get("month", 7.0))
        rec["day_of_year"] = float(rec.get("day_of_year", 196.0))
        rec["lead_time_hours"] = float(rec.get("lead_time_hours", 24.0))
        return rec

    df = df_or_record.copy()
    rain = np.maximum(0.0, df["rainfall_nwp"] if "rainfall_nwp" in df.columns else 0.0)
    df["rainfall_nwp"] = rain
    df["log_rainfall_nwp"] = np.log1p(rain)
    
    rh = df["humidity"] if "humidity" in df.columns else 75.0
    temp = df["temperature"] if "temperature" in df.columns else 26.0
    df["dewpoint_approx"] = temp - ((100.0 - rh) / 5.0)
    
    if "wind_u" not in df.columns:
        df["wind_u"] = df.get("wind_speed", 10.0) * 0.8
    if "wind_v" not in df.columns:
        df["wind_v"] = df.get("wind_speed", 10.0) * 0.6
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
        
    df["moisture_flux"] = df["wind_speed"] * (rh / 100.0)
    
    elev = df["elevation"] if "elevation" in df.columns else 0.0
    omega = df["vertical_velocity"] if "vertical_velocity" in df.columns else -0.2
    df["orographic_enhancement_index"] = (elev / 500.0) * np.maximum(0.0, -omega * 10.0)
    
    lat = df["lat"] if "lat" in df.columns else 20.0
    lon = df["lon"] if "lon" in df.columns else 78.0
    df["is_monsoon_core"] = ((lat >= 18.0) & (lat <= 26.0) & (lon >= 74.0) & (lon <= 86.0)).astype(float)
    
    if "lead_time_hours" not in df.columns:
        df["lead_time_hours"] = 24.0
    if "month" not in df.columns:
        df["month"] = 7.0
    if "day_of_year" not in df.columns:
        df["day_of_year"] = 196.0

    return df

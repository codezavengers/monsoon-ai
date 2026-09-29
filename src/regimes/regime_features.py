"""
Feature engineering specific to regime identification and atmospheric dynamics.
"""

from typing import Dict, Any, List
import numpy as np

def compute_regime_features(df_or_dict: Any) -> Any:
    """Computes dynamic atmospheric indicators for regime classification."""
    is_dict = isinstance(df_or_dict, dict)
    
    if is_dict:
        d = dict(df_or_dict)
        rain = float(d.get("rainfall_nwp", d.get("rainfall", 0.0)))
        d["log_rainfall_nwp"] = float(np.log1p(max(0.0, rain)))
        u = float(d.get("wind_u", 0.0))
        v = float(d.get("wind_v", 0.0))
        if "wind_speed" not in d or d["wind_speed"] == 0:
            d["wind_speed"] = float(np.sqrt(u**2 + v**2))
        
        # Moisture flux proxy = wind_speed * (humidity / 100)
        rh = float(d.get("humidity", 70.0))
        d["moisture_flux"] = float(d["wind_speed"] * (rh / 100.0))
        
        # Orographic index = elevation * max(0, -vertical_velocity or wind_speed)
        elev = float(d.get("elevation", 0.0))
        omega = float(d.get("vertical_velocity", -0.1))
        d["orographic_enhancement_index"] = float((elev / 500.0) * max(0.0, -omega * 10.0))
        
        # Dewpoint approximation (Magnus-Tetens)
        temp = float(d.get("temperature", 25.0))
        d["dewpoint_approx"] = float(temp - ((100.0 - rh) / 5.0))
        
        lat = float(d.get("lat", d.get("latitude", 20.0)))
        lon = float(d.get("lon", d.get("longitude", 78.0)))
        d["is_monsoon_core"] = 1.0 if (18.0 <= lat <= 26.0 and 74.0 <= lon <= 86.0) else 0.0
        return d

    # DataFrame branch
    df = df_or_dict.copy()
    if "rainfall_nwp" in df.columns:
        df["log_rainfall_nwp"] = np.log1p(np.maximum(0.0, df["rainfall_nwp"]))
    
    if "wind_speed" not in df.columns and "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
    elif "wind_speed" not in df.columns:
        df["wind_speed"] = 10.0
        
    rh = df["humidity"] if "humidity" in df.columns else 75.0
    df["moisture_flux"] = df["wind_speed"] * (rh / 100.0)
    
    elev = df["elevation"] if "elevation" in df.columns else 0.0
    omega = df["vertical_velocity"] if "vertical_velocity" in df.columns else -0.2
    df["orographic_enhancement_index"] = (elev / 500.0) * np.maximum(0.0, -omega * 10.0)
    
    temp = df["temperature"] if "temperature" in df.columns else 26.0
    df["dewpoint_approx"] = temp - ((100.0 - rh) / 5.0)
    
    lat = df["lat"] if "lat" in df.columns else 20.0
    lon = df["lon"] if "lon" in df.columns else 78.0
    df["is_monsoon_core"] = ((lat >= 18.0) & (lat <= 26.0) & (lon >= 74.0) & (lon <= 86.0)).astype(float)
    
    return df

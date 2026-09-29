"""
Rule-Based Weather Regime Classification for the Indian Monsoon.
Defines canonical synoptic and meso-scale meteorological weather regimes:
1. Normal / Background Monsoon
2. Active Monsoon
3. Break Monsoon
4. Monsoon Depression
5. Coastal Rainfall
6. Orographic Rainfall
7. Western Disturbance
8. Extreme Convective Event
"""

from typing import Dict, Any, Union
import numpy as np

REGIME_NAMES = [
    "normal_monsoon",
    "active_monsoon",
    "break_monsoon",
    "monsoon_depression",
    "coastal_rainfall",
    "orographic_rainfall",
    "western_disturbance",
    "extreme_event"
]

REGIME_DISPLAY_NAMES = {
    "normal_monsoon": "Normal / Background Monsoon",
    "active_monsoon": "Active Monsoon",
    "break_monsoon": "Break Monsoon",
    "monsoon_depression": "Monsoon Low / Depression",
    "coastal_rainfall": "Coastal Rainfall",
    "orographic_rainfall": "Orographic Rainfall",
    "western_disturbance": "Western Disturbance",
    "extreme_event": "Extreme Convective Event"
}

def classify_regime_rule(record: Union[Dict[str, Any], Any]) -> str:
    """
    Classifies a meteorological record into one of 8 weather regimes using domain physics.
    Supports dictionary or pandas Series / row objects.
    """
    def _get(key, default=0.0):
        if isinstance(record, dict):
            return record.get(key, default)
        if hasattr(record, "get"):
            return record.get(key, default)
        try:
            return record[key]
        except (KeyError, IndexError, TypeError):
            return default

    rainfall = float(_get("rainfall_nwp", _get("rainfall", 0.0)))
    pressure = float(_get("pressure", _get("prmsl", 1004.0)))
    wind_speed = float(_get("wind_speed", 10.0))
    humidity = float(_get("humidity", _get("rh2m", 75.0)))
    cape = float(_get("cape", 1500.0))
    elevation = float(_get("elevation", 50.0))
    coast_dist_km = float(_get("coast_dist_km", 100.0))
    latitude = float(_get("latitude", _get("lat", 20.0)))
    longitude = float(_get("longitude", _get("lon", 78.0)))
    vertical_velocity = float(_get("vertical_velocity", _get("vvel500", -0.2)))

    # 1. Extreme Convective Event (IMD extreme > 100mm, or heavy + deep convection)
    if rainfall >= 100.0 or (rainfall >= 60.0 and cape > 2400.0 and vertical_velocity < -0.4):
        return "extreme_event"

    # 2. Monsoon Depression (Deep barometric depression <= 998 hPa + cyclonic winds)
    if pressure <= 998.0 and wind_speed >= 12.0 and rainfall >= 35.0:
        return "monsoon_depression"

    # 3. Orographic Rainfall (Western Ghats windward slopes & Himalayan foothills)
    is_mountainous = elevation >= 500.0 or ((latitude < 20.0 and longitude < 76.5) and elevation >= 250.0)
    if is_mountainous and rainfall >= 25.0 and humidity >= 80.0:
        return "orographic_rainfall"

    # 4. Coastal Rainfall (Narrow coastal boundary with land-sea moisture convergence)
    if coast_dist_km <= 35.0 and rainfall >= 20.0 and humidity >= 75.0:
        return "coastal_rainfall"

    # 5. Western Disturbance (Upper-air trough in northwest India: lat >= 28N, lon <= 78E)
    if latitude >= 28.0 and longitude <= 78.0 and rainfall >= 10.0 and pressure <= 1005.0:
        return "western_disturbance"

    # 6. Break Monsoon (Suppressed rainfall < 5mm and dry air over core monsoon zone 18-26N, 74-86E)
    is_monsoon_core = (18.0 <= latitude <= 26.0) and (74.0 <= longitude <= 86.0)
    if is_monsoon_core and rainfall < 5.0 and humidity < 65.0:
        return "break_monsoon"

    # 7. Active Monsoon (Widespread monsoon rainfall with strong southwesterlies)
    if rainfall >= 25.0 and humidity >= 75.0 and wind_speed >= 8.0:
        return "active_monsoon"

    # 8. Normal / Background Monsoon
    return "normal_monsoon"

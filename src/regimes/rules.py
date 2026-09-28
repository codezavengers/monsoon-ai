"""
Rule-Based Meteorological Weather Regime Classifier.
Prototypes rules representing synoptic and mesoscale conditions during Indian Monsoon.
"""

from typing import Dict, Any, List

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

def classify_regime_rule(row: Dict[str, Any]) -> str:
    """
    Classifies a meteorological record into one of 8 prototype weather regimes.
    
    Variables considered:
    - rainfall_nwp: NWP forecast rainfall (mm)
    - pressure: Mean sea level pressure (hPa)
    - wind_speed: Wind speed (m/s)
    - humidity: Relative humidity (%)
    - cape: Convective Available Potential Energy (J/kg)
    - elevation: Elevation above sea level (m)
    - coast_dist_km: Distance from coastline (km)
    - lat: Latitude (°N)
    - lon: Longitude (°E)
    - vertical_velocity: Omega / updraft (Pa/s, negative indicates ascent)
    """
    rainfall = float(row.get("rainfall_nwp", 0.0))
    pressure = float(row.get("pressure", 1005.0))
    wind = float(row.get("wind_speed", 7.0))
    rh = float(row.get("humidity", 75.0))
    cape = float(row.get("cape", 1200.0))
    elevation = float(row.get("elevation", 100.0))
    coast_dist = float(row.get("coast_dist_km", 200.0))
    lat = float(row.get("lat", 20.0))
    lon = float(row.get("lon", 80.0))
    omega = float(row.get("vertical_velocity", -0.2)) # Pa/s (< 0 is updraft)
    
    # 1. Extreme Convective Event: extreme precipitation or extreme CAPE with strong ascent
    if rainfall >= 100.0 or (rainfall >= 60.0 and cape > 2400 and omega < -0.4):
        return "extreme_event"
        
    # 2. Monsoon Low / Depression: Low sea-level pressure (< 998 hPa), strong cyclonic winds, heavy rain
    if pressure <= 998.0 and wind >= 12.0 and rainfall >= 35.0:
        return "monsoon_depression"
        
    # 3. Orographic Rainfall: High elevation / Ghats / Himalayas with high moisture & rainfall
    # Western Ghats or NE Hills / Himalayan foothills
    is_mountainous = elevation >= 500.0 or ((lat < 20.0 and lon < 76.5) and elevation >= 250.0)
    if is_mountainous and rainfall >= 25.0 and rh >= 80.0:
        return "orographic_rainfall"
        
    # 4. Coastal Rainfall: Near coastline (< 35 km), high moisture, coastal convergence
    if coast_dist <= 35.0 and rh >= 82.0 and rainfall >= 15.0:
        return "coastal_rainfall"
        
    # 5. Western Disturbance: Northern India (lat >= 28.0), moderate/high elevation, characteristic mid-latitude trough
    if lat >= 28.0 and pressure <= 1004.0 and (wind >= 9.0 or elevation > 600) and rainfall >= 10.0:
        return "western_disturbance"
        
    # 6. Active Monsoon: High widespread moisture, widespread rain across core monsoon zone, vigorous monsoon trough
    # Core monsoon zone: lat 18-26, lon 73-88
    in_core_zone = (18.0 <= lat <= 26.0) and (73.0 <= lon <= 88.0)
    if (in_core_zone or rh >= 82.0) and rainfall >= 25.0 and rh >= 80.0 and wind >= 8.0:
        return "active_monsoon"
        
    # 7. Break Monsoon: Core monsoon zone has very suppressed rainfall and low humidity
    if in_core_zone and rainfall < 5.0 and rh < 68.0 and pressure >= 1008.0:
        return "break_monsoon"
        
    # 8. Background / Normal Monsoon: Default synoptic state
    return "normal_monsoon"

def batch_classify_rules(records: List[Dict[str, Any]]) -> List[str]:
    """Classifies a list of meteorological records using the prototype rules."""
    return [classify_regime_rule(r) for r in records]

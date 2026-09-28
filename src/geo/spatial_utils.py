"""
Spatial utilities for meteorological coordinates, distances, neighborhood extraction, and aggregations.
"""

import math
from typing import Tuple, List, Dict, Any
import numpy as np

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in km."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_nearest_district(lat: float, lon: float, districts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds the nearest district from a reference list given lat/lon."""
    min_dist = float("inf")
    nearest = districts[0]
    for d in districts:
        dist = haversine_distance_km(lat, lon, d["lat"], d["lon"])
        if dist < min_dist:
            min_dist = dist
            nearest = d
    return nearest

def aggregate_grid_to_districts(
    grid_data: List[Dict[str, Any]], 
    districts: List[Dict[str, Any]], 
    radius_km: float = 75.0
) -> List[Dict[str, Any]]:
    """
    Aggregates gridded meteorological forecasts to district level.
    Computes mean, max, 75th percentile, and 90th percentile.
    """
    district_results = []
    for dist in districts:
        dlat, dlon = dist["lat"], dist["lon"]
        # Match grid points within radius
        matched_points = [
            pt for pt in grid_data
            if haversine_distance_km(dlat, dlon, pt["lat"], pt["lon"]) <= radius_km
        ]
        
        # Fallback to closest point if none within radius
        if not matched_points:
            closest = min(grid_data, key=lambda pt: haversine_distance_km(dlat, dlon, pt["lat"], pt["lon"]))
            matched_points = [closest]
            
        raw_vals = [pt.get("rainfall_nwp", 0.0) for pt in matched_points]
        corrected_vals = [pt.get("rainfall_corrected", pt.get("rainfall_nwp", 0.0)) for pt in matched_points]
        obs_vals = [pt.get("rainfall_obs", 0.0) for pt in matched_points]
        p_heavy_vals = [pt.get("p_heavy", 0.0) for pt in matched_points]
        p_very_heavy_vals = [pt.get("p_very_heavy", 0.0) for pt in matched_points]
        p_extreme_vals = [pt.get("p_extreme", 0.0) for pt in matched_points]
        
        # Majority regime
        regimes = [pt.get("regime", "normal_monsoon") for pt in matched_points]
        dominant_regime = max(set(regimes), key=regimes.count) if regimes else "normal_monsoon"
        
        raw_mean = float(np.mean(raw_vals))
        raw_max = float(np.max(raw_vals))
        corr_mean = float(np.mean(corrected_vals))
        corr_max = float(np.max(corrected_vals))
        corr_p90 = float(np.percentile(corrected_vals, 90))
        obs_mean = float(np.mean(obs_vals)) if obs_vals else 0.0
        
        # Categorize rainfall according to IMD standards
        if corr_max >= 204.5:
            cat = "Extremely Heavy"
        elif corr_max >= 115.6:
            cat = "Very Heavy"
        elif corr_max >= 64.5:
            cat = "Heavy"
        elif corr_max >= 15.6:
            cat = "Moderate"
        elif corr_max >= 2.5:
            cat = "Light"
        else:
            cat = "No Rain"
            
        district_results.append({
            "district": dist["district"],
            "state": dist["state"],
            "lat": dist["lat"],
            "lon": dist["lon"],
            "elevation": dist["elevation"],
            "coast_dist_km": dist["coast_dist_km"],
            "zone": dist["zone"],
            "regime": dominant_regime,
            "raw_nwp_mean": round(raw_mean, 1),
            "raw_nwp_max": round(raw_max, 1),
            "corrected_mean": round(corr_mean, 1),
            "corrected_max": round(corr_max, 1),
            "corrected_p90": round(corr_p90, 1),
            "observed_mean": round(obs_mean, 1),
            "delta_correction": round(corr_mean - raw_mean, 1),
            "p_heavy": round(float(np.mean(p_heavy_vals)), 3),
            "p_very_heavy": round(float(np.mean(p_very_heavy_vals)), 3),
            "p_extreme": round(float(np.mean(p_extreme_vals)), 3),
            "category": cat
        })
        
    return district_results

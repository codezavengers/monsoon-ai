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

def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon check."""
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1] + 1e-12) + poly[i][0]):
            c = not c
        j = i
    return c

def aggregate_to_states(district_forecasts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates district-level forecasts to state level.
    Computes state-level mean, max, and high-impact district counts.
    """
    state_groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in district_forecasts:
        st = d.get("state", "Other")
        state_groups.setdefault(st, []).append(d)

    state_results = {}
    for st, dists in state_groups.items():
        raw_means = [d.get("raw_nwp_mean", 0.0) for d in dists]
        corr_means = [d.get("corrected_mean", 0.0) for d in dists]
        corr_maxs = [d.get("corrected_max", 0.0) for d in dists]
        heavy_count = sum(1 for d in dists if d.get("corrected_max", 0.0) >= 64.5)
        very_heavy_count = sum(1 for d in dists if d.get("corrected_max", 0.0) >= 115.6)

        state_results[st] = {
            "state": st,
            "district_count": len(dists),
            "raw_nwp_mean": round(float(np.mean(raw_means)), 1),
            "corrected_mean": round(float(np.mean(corr_means)), 1),
            "max_rainfall": round(float(np.max(corr_maxs)), 1),
            "heavy_rain_districts": heavy_count,
            "very_heavy_districts": very_heavy_count
        }
    return state_results

def aggregate_to_meteorological_regions(district_forecasts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates forecasts to IMD 4 broad meteorological regions:
    - Northwest India
    - Central India
    - East & Northeast India
    - South Peninsular India
    """
    region_mapping = {
        "Western Ghats": "South Peninsular India",
        "South Interior Karnataka": "South Peninsular India",
        "Tamil Nadu Coast": "South Peninsular India",
        "Tamil Nadu Interior": "South Peninsular India",
        "Telangana Plateau": "South Peninsular India",
        "Coastal Andhra": "South Peninsular India",
        "Kerala Coast": "South Peninsular India",
        "Central India Core": "Central India",
        "Vidarbha": "Central India",
        "Marathwada": "Central India",
        "North Konkan": "Central India",
        "South Konkan": "Central India",
        "Gujarat Plains": "Central India",
        "Saurashtra Coast": "Central India",
        "East Madhya Pradesh": "Central India",
        "West Madhya Pradesh": "Central India",
        "East Rajasthan": "Northwest India",
        "West Rajasthan": "Northwest India",
        "Haryana Plains": "Northwest India",
        "Punjab": "Northwest India",
        "Himachal Hills": "Northwest India",
        "Kashmir Valley": "Northwest India",
        "Uttarakhand Foothills": "Northwest India",
        "West UP Plains": "Northwest India",
        "East UP Plains": "Central India",
        "North Bihar": "East & Northeast India",
        "South Bihar": "East & Northeast India",
        "Gangetic West Bengal": "East & Northeast India",
        "Sub-Himalayan Bengal": "East & Northeast India",
        "Brahmaputra Valley": "East & Northeast India",
        "Barak Valley": "East & Northeast India",
        "Meghalaya Plateau": "East & Northeast India",
        "Odisha Coast": "East & Northeast India",
        "Interior Odisha": "East & Northeast India"
    }

    region_groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in district_forecasts:
        zone = d.get("zone", "")
        region = region_mapping.get(zone, "Central India")
        region_groups.setdefault(region, []).append(d)

    results = {}
    for reg_name, dists in region_groups.items():
        raw_m = [d.get("raw_nwp_mean", 0.0) for d in dists]
        corr_m = [d.get("corrected_mean", 0.0) for d in dists]
        corr_max = [d.get("corrected_max", 0.0) for d in dists]
        results[reg_name] = {
            "region": reg_name,
            "district_count": len(dists),
            "raw_nwp_mean": round(float(np.mean(raw_m)), 1),
            "corrected_mean": round(float(np.mean(corr_m)), 1),
            "regional_max": round(float(np.max(corr_max)), 1),
            "delta_correction": round(float(np.mean(corr_m) - np.mean(raw_m)), 1)
        }
    return results

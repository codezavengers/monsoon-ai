"""
Geospatial math and spatial verification distance utilities.
"""

from typing import Tuple
import numpy as np

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)

def compute_centroid_displacement(
    grid_forecast: np.ndarray,
    grid_observed: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold_mm: float = 15.0
) -> Tuple[float, float, float]:
    """
    Computes precipitation centroid displacement between forecast and observed fields.
    Returns: (delta_lat_deg, delta_lon_deg, displacement_distance_km)
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    fc_mask = grid_forecast >= threshold_mm
    obs_mask = grid_observed >= threshold_mm
    
    if np.sum(fc_mask) == 0 or np.sum(obs_mask) == 0:
        return 0.0, 0.0, 0.0
        
    fc_lat_c = np.sum(lat_grid[fc_mask] * grid_forecast[fc_mask]) / np.sum(grid_forecast[fc_mask])
    fc_lon_c = np.sum(lon_grid[fc_mask] * grid_forecast[fc_mask]) / np.sum(grid_forecast[fc_mask])
    
    obs_lat_c = np.sum(lat_grid[obs_mask] * grid_observed[obs_mask]) / np.sum(grid_observed[obs_mask])
    obs_lon_c = np.sum(lon_grid[obs_mask] * grid_observed[obs_mask]) / np.sum(grid_observed[obs_mask])
    
    delta_lat = float(obs_lat_c - fc_lat_c)
    delta_lon = float(obs_lon_c - fc_lon_c)
    dist_km = haversine_km(fc_lat_c, fc_lon_c, obs_lat_c, obs_lon_c)
    
    return delta_lat, delta_lon, dist_km

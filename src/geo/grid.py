"""
Geographic grid operations and grid-to-district spatial aggregations.
"""

from typing import Dict, Any, Tuple
import numpy as np
from .district_mapping import INDIAN_DISTRICTS

def get_india_grid(resolution_deg: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Returns India domain regular grid coordinates [6-38N, 68-98E]."""
    lats = np.arange(6.0, 38.0 + resolution_deg, resolution_deg)
    lons = np.arange(68.0, 98.0 + resolution_deg, resolution_deg)
    return lats, lons

def aggregate_2d_grid_to_districts(
    grid_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Maps a 2D rainfall field onto Indian districts using station coordinates."""
    results = {}
    for district, info in INDIAN_DISTRICTS.items():
        lat_c = info["lat"]
        lon_c = info["lon"]
        lat_idx = int(np.argmin(np.abs(lats - lat_c)))
        lon_idx = int(np.argmin(np.abs(lons - lon_c)))
        
        y0 = max(0, lat_idx - 1)
        y1 = min(len(lats), lat_idx + 2)
        x0 = max(0, lon_idx - 1)
        x1 = min(len(lons), lon_idx + 2)
        
        vals = grid_2d[y0:y1, x0:x1].flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            vals = np.array([0.0])
            
        results[district] = {
            "mean": round(float(np.mean(vals)), 1),
            "max": round(float(np.max(vals)), 1),
            "p90": round(float(np.percentile(vals, 90)), 1),
            "min": round(float(np.min(vals)), 1),
            "aggregation_method": "REPRESENTATIVE_POINT_RADIUS"
        }
    return results

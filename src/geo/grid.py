"""
2D Gridded Spatial Domain and Geography for India.
Defines standard IMD / NWP 0.25° and 0.5° coordinate grids, land masks,
and spatial aggregation to districts and states.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from src.geo.spatial_utils import haversine_distance_km

# India Subcontinent geographic boundary polygon approximations for land-sea masking
# Coarse polygon vertices bounding Indian landmass
INDIA_LAND_VERTICES = [
    (8.0, 77.5), (8.5, 76.9), (10.0, 75.8), (12.0, 75.0), (14.5, 74.3),
    (16.0, 73.3), (18.9, 72.8), (20.5, 72.8), (22.0, 69.0), (23.5, 68.5),
    (24.5, 71.0), (27.0, 70.0), (30.0, 73.0), (32.5, 74.5), (35.0, 74.5),
    (36.5, 77.0), (34.5, 78.5), (32.0, 79.0), (30.5, 81.0), (28.0, 82.0),
    (27.5, 88.0), (27.0, 89.0), (28.0, 96.0), (27.0, 97.0), (24.0, 94.0),
    (22.0, 92.5), (21.5, 87.0), (19.5, 85.0), (17.5, 83.0), (15.5, 80.0),
    (13.0, 80.2), (11.0, 79.8), (9.0, 78.5), (8.0, 77.5)
]

def generate_india_grid(resolution_deg: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates regular 2D grid across Indian subcontinent [8.0°N to 36.0°N, 68.5°E to 97.0°E].
    Returns:
      lats (1D), lons (1D), land_mask (2D boolean array where True is Indian landmass)
    """
    lats = np.arange(8.0, 36.0 + resolution_deg / 2, resolution_deg)
    lons = np.arange(68.5, 97.0 + resolution_deg / 2, resolution_deg)

    ny, nx = len(lats), len(lons)
    land_mask = np.zeros((ny, nx), dtype=bool)

    # Simplified bounding box polygon containment
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Peninsular South India narrowing
            if lat < 20.0:
                # Longitude range between West Coast and East Coast
                w_lon = 72.0 + (20.0 - lat) * 0.35
                e_lon = 85.0 - (20.0 - lat) * 0.45
                if w_lon <= lon <= e_lon:
                    land_mask[i, j] = True
            # Central & Northern India
            elif 20.0 <= lat <= 32.0:
                if 69.5 <= lon <= 96.0:
                    land_mask[i, j] = True
            # Himalayan & Northern reach
            elif lat > 32.0:
                if 73.5 <= lon <= 80.5:
                    land_mask[i, j] = True

    return lats, lons, land_mask

def aggregate_2d_grid_to_districts(
    grid_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    districts: List[Dict[str, Any]],
    radius_km: float = 65.0
) -> Dict[str, Dict[str, float]]:
    """
    Aggregates genuine 2-D gridded field to district summary statistics.
    Returns dict: district_name -> {mean, max, p90, min}
    """
    ny, nx = len(lats), len(lons)
    mesh_lats, mesh_lons = np.meshgrid(lats, lons, indexing="ij")
    results = {}

    for d in districts:
        dlat, dlon = d["lat"], d["lon"]
        # Fast bounding box pre-filter (~1 degree is ~111 km)
        deg_radius = (radius_km / 111.0) * 1.5
        box_mask = (np.abs(mesh_lats - dlat) <= deg_radius) & (np.abs(mesh_lons - dlon) <= deg_radius)

        box_indices = np.where(box_mask)
        matched_vals = []

        if len(box_indices[0]) > 0:
            for y_idx, x_idx in zip(box_indices[0], box_indices[1]):
                dist_km = haversine_distance_km(dlat, dlon, lats[y_idx], lons[x_idx])
                if dist_km <= radius_km:
                    v = grid_2d[y_idx, x_idx]
                    if not np.isnan(v):
                        matched_vals.append(v)

        if not matched_vals:
            # Nearest grid cell fallback
            y_near = int(np.argmin(np.abs(lats - dlat)))
            x_near = int(np.argmin(np.abs(lons - dlon)))
            matched_vals = [grid_2d[y_near, x_near]]

        arr = np.array(matched_vals, dtype=float)
        results[d["district"]] = {
            "mean": round(float(np.mean(arr)), 1),
            "max": round(float(np.max(arr)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "min": round(float(np.min(arr)), 1)
        }

    return results

def get_geojson_feature_collection(districts_forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds GeoJSON FeatureCollection representation of district forecasts
    suitable for mapping and GIS interoperability.
    """
    features = []
    for d in districts_forecast:
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [d["lon"], d["lat"]]
            },
            "properties": {
                "district": d["district"],
                "state": d["state"],
                "zone": d["zone"],
                "regime": d["regime"],
                "raw_nwp_max": d.get("raw_nwp_max", 0.0),
                "corrected_max": d.get("corrected_max", 0.0),
                "delta_correction": d.get("delta_correction", 0.0),
                "p_heavy": d.get("p_heavy", 0.0),
                "p_very_heavy": d.get("p_very_heavy", 0.0),
                "p_extreme": d.get("p_extreme", 0.0),
                "category": d.get("category", "Moderate")
            }
        }
        features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features
    }

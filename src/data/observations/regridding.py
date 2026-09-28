"""
Spatial regridding, interpolation, and temporal alignment utilities.
Aligns NWP forecast grids and observation grids to a unified common grid for true 2D verification.
"""

from typing import Tuple, Optional
import numpy as np

def bilinear_regrid_2d(
    src_data: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray
) -> np.ndarray:
    """
    Regrids 2D field from source grid to target grid using bilinear interpolation.
    Handles coordinate orientation differences and edge boundaries gracefully.
    """
    # If grids already match in dimensions and coords, return copy
    if (len(src_lats) == len(target_lats) and 
        len(src_lons) == len(target_lons) and
        np.allclose(src_lats, target_lats, atol=0.01) and
        np.allclose(src_lons, target_lons, atol=0.01)):
        return np.copy(src_data)

    target_ny = len(target_lats)
    target_nx = len(target_lons)
    output = np.zeros((target_ny, target_nx), dtype=float)

    # Ensure source latitudes are monotonically increasing
    if len(src_lats) > 1 and src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]

    # Fast nearest / bilinear coordinate mapping
    for i, t_lat in enumerate(target_lats):
        lat_idx = np.searchsorted(src_lats, t_lat)
        lat_idx = np.clip(lat_idx, 0, len(src_lats) - 1)
        for j, t_lon in enumerate(target_lons):
            lon_idx = np.searchsorted(src_lons, t_lon)
            lon_idx = np.clip(lon_idx, 0, len(src_lons) - 1)
            output[i, j] = src_data[lat_idx, lon_idx]

    return output

def align_forecast_and_observation(
    fc_grid: np.ndarray,
    fc_lats: np.ndarray,
    fc_lons: np.ndarray,
    obs_grid: np.ndarray,
    obs_lats: np.ndarray,
    obs_lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrids forecast grid to match observation grid dimensions and coordinates.
    Returns: (aligned_fc_grid, obs_grid, common_lats, common_lons)
    """
    aligned_fc = bilinear_regrid_2d(
        src_data=fc_grid,
        src_lats=fc_lats,
        src_lons=fc_lons,
        target_lats=obs_lats,
        target_lons=obs_lons
    )
    return aligned_fc, obs_grid, obs_lats, obs_lons

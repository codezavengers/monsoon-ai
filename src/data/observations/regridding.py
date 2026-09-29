"""
Spatial Regridding Utilities for Aligning NWP Grids to IMD 0.25° Grids.
"""

from typing import Tuple
import numpy as np
import xarray as xr

def regrid_to_target_grid(
    source_ds: xr.Dataset,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    method: str = "linear"
) -> xr.Dataset:
    """Interpolates source dataset to match target latitude/longitude coordinates."""
    return source_ds.interp(lat=target_lats, lon=target_lons, method=method)

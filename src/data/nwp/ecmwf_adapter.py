"""
ECMWF (European Centre for Medium-Range Weather Forecasts) Integrated Forecasting System (IFS/HRES) Adapter.
Handles ECMWF 0.1° / 0.25° gridded outputs.
Standardizes ECMWF GRIB parameters (total precipitation 'tp' in meters to mm/day, 2t, 2r, msl, cape, w).
"""

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np

from src.data.nwp.base import BaseNWPAdapter, NWPValidationError, INDIA_BBOX

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

class ECMWFAdapter(BaseNWPAdapter):
    """
    Adapter for ECMWF HRES/IFS 0.25-degree forecast products.
    ECMWF initialized at 00Z and 12Z.
    """
    def __init__(self):
        super().__init__(provider_name="ECMWF_HRES_0.25deg", resolution_deg=0.25)

    def load_data(
        self,
        source_path: str,
        lead_time_hours: int = 24,
        init_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """Loads ECMWF dataset from NetCDF, GRIB2 or structured fallback."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"ECMWF data file not found at: {source_path}")

        ext = os.path.splitext(source_path)[-1].lower()
        if ext in [".nc", ".nc4", ".grb2", ".grib2"] and XARRAY_AVAILABLE:
            return self._load_from_xarray(source_path, lead_time_hours, init_time)
        else:
            return self._load_fallback(source_path, lead_time_hours, init_time)

    def _load_from_xarray(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime]
    ) -> Dict[str, Any]:
        ds = xr.open_dataset(filepath)

        lat_coord = next((c for c in ["latitude", "lat"] if c in ds.coords), None)
        lon_coord = next((c for c in ["longitude", "lon"] if c in ds.coords), None)

        if not lat_coord or not lon_coord:
            raise NWPValidationError(f"Could not find coordinates in ECMWF file: {filepath}")

        lats = ds[lat_coord].values
        lons = self.normalize_longitudes(ds[lon_coord].values)

        # ECMWF lats are often in descending order (North to South); ensure ascending or handle slice
        lat_mask = (lats >= INDIA_BBOX["lat_min"] - 0.5) & (lats <= INDIA_BBOX["lat_max"] + 0.5)
        lon_mask = (lons >= INDIA_BBOX["lon_min"] - 0.5) & (lons <= INDIA_BBOX["lon_max"] + 0.5)

        sub_lats = lats[lat_mask]
        sub_lons = lons[lon_mask]
        self.validate_spatial_domain(sub_lats, sub_lons)

        var_mapping = {
            "rainfall_nwp": ["tp", "total_precipitation", "cp"],
            "temperature": ["2t", "t2m"],
            "humidity": ["r", "rh"],
            "pressure": ["msl", "sp"],
            "cape": ["cape"],
            "vertical_velocity": ["w", "omega"]
        }

        canonical_vars = {}
        for canonical, candidates in var_mapping.items():
            for cand in candidates:
                if cand in ds.variables:
                    data = ds[cand].values
                    if data.ndim == 3:
                        data = data[-1]
                    canonical_vars[canonical] = data[np.ix_(lat_mask, lon_mask)]
                    break

        # ECMWF 'tp' is in meters! Convert to millimeters: 1 m = 1000 mm
        if "rainfall_nwp" in canonical_vars:
            raw_tp = canonical_vars["rainfall_nwp"]
            if np.nanmax(raw_tp) < 2.0: # If in meters (e.g. 0.050 m)
                canonical_vars["rainfall_nwp"] = raw_tp * 1000.0

        warnings = self.validate_variables(canonical_vars)

        now = init_time or datetime.datetime.utcnow()
        valid = now + datetime.timedelta(hours=lead_time_hours)

        return {
            "provider": self.provider_name,
            "grid_lats": sub_lats,
            "grid_lons": sub_lons,
            "variables": canonical_vars,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "init_time": now.isoformat(),
                "valid_time": valid.isoformat(),
                "lead_time_hours": lead_time_hours,
                "warnings": warnings,
                "spatial_resolution": f"{self.resolution_deg}° x {self.resolution_deg}°"
            }
        }

    def _load_fallback(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime]
    ) -> Dict[str, Any]:
        """Generic fallback for JSON or CSV."""
        from src.data.nwp.gfs_adapter import GFSAdapter
        gfs = GFSAdapter()
        res = gfs.load_data(filepath, lead_time_hours, init_time)
        res["provider"] = self.provider_name
        return res

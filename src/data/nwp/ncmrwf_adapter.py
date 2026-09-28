"""
NCMRWF (National Centre for Medium Range Weather Forecasting, Ministry of Earth Sciences, Govt of India) Adapter.
Handles NCUM (NCMRWF Unified Model ~12 km), NEPS (NCMRWF Ensemble Prediction System), and GFS-T1534 operational models.
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

class NCMRWFAdapter(BaseNWPAdapter):
    """
    Adapter for MoES / NCMRWF operational forecast outputs.
    Optimized for Indian Monsoon domain [6-38°N, 68-98°E].
    """
    def __init__(self):
        super().__init__(provider_name="NCMRWF_NCUM_12km", resolution_deg=0.12)

    def load_data(
        self,
        source_path: str,
        lead_time_hours: int = 24,
        init_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"NCMRWF data file not found at: {source_path}")

        ext = os.path.splitext(source_path)[-1].lower()
        if ext in [".nc", ".nc4", ".grb2", ".grib2"] and XARRAY_AVAILABLE:
            ds = xr.open_dataset(source_path)
            lat_coord = next((c for c in ["latitude", "lat"] if c in ds.coords), None)
            lon_coord = next((c for c in ["longitude", "lon"] if c in ds.coords), None)

            if not lat_coord or not lon_coord:
                raise NWPValidationError(f"Could not find coordinates in NCMRWF file: {source_path}")

            lats = ds[lat_coord].values
            lons = self.normalize_longitudes(ds[lon_coord].values)

            lat_mask = (lats >= INDIA_BBOX["lat_min"] - 0.5) & (lats <= INDIA_BBOX["lat_max"] + 0.5)
            lon_mask = (lons >= INDIA_BBOX["lon_min"] - 0.5) & (lons <= INDIA_BBOX["lon_max"] + 0.5)

            sub_lats = lats[lat_mask]
            sub_lons = lons[lon_mask]
            self.validate_spatial_domain(sub_lats, sub_lons)

            var_mapping = {
                "rainfall_nwp": ["precip", "tot_prec", "precipitation_flux", "rain"],
                "temperature": ["temp", "t2m", "air_temperature"],
                "humidity": ["rh", "relative_humidity"],
                "pressure": ["mslp", "prmsl", "surface_air_pressure"],
                "cape": ["cape", "convective_available_potential_energy"]
            }

            canonical_vars = {}
            for canonical, candidates in var_mapping.items():
                for cand in candidates:
                    if cand in ds.variables:
                        data = ds[cand].values
                        if data.ndim >= 3:
                            data = data[-1]
                        canonical_vars[canonical] = data[np.ix_(lat_mask, lon_mask)]
                        break

            # Handle precipitation flux (kg m-2 s-1) to mm/day (multiply by 86400)
            if "rainfall_nwp" in canonical_vars:
                p_val = canonical_vars["rainfall_nwp"]
                if np.nanmax(p_val) < 0.05: # In flux units (e.g. 0.0002 kg/m2/s)
                    canonical_vars["rainfall_nwp"] = p_val * 86400.0

            warnings = self.validate_variables(canonical_vars)

            now = init_time or datetime.datetime.utcnow()
            valid = now + datetime.timedelta(hours=lead_time_hours)

            return {
                "provider": self.provider_name,
                "grid_lats": sub_lats,
                "grid_lons": sub_lons,
                "variables": canonical_vars,
                "metadata": {
                    "source_file": os.path.basename(source_path),
                    "init_time": now.isoformat(),
                    "valid_time": valid.isoformat(),
                    "lead_time_hours": lead_time_hours,
                    "warnings": warnings,
                    "spatial_resolution": f"{self.resolution_deg}° x {self.resolution_deg}°"
                }
            }
        else:
            from src.data.nwp.gfs_adapter import GFSAdapter
            gfs = GFSAdapter()
            res = gfs.load_data(source_path, lead_time_hours, init_time)
            res["provider"] = self.provider_name
            return res

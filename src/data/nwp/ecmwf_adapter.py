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
        init_time: Optional[datetime.datetime] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Loads ECMWF dataset from NetCDF, GRIB2 or provider-native format."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"ECMWF data file not found at: {source_path}")

        current_mode = mode or os.environ.get("MODE", "DEMO")
        self.validate_data_mode_path(source_path, mode=current_mode)

        ext = os.path.splitext(source_path)[-1].lower()
        if ext in [".nc", ".nc4", ".grb2", ".grib2"] and XARRAY_AVAILABLE:
            return self._load_from_xarray(source_path, lead_time_hours, init_time, mode=current_mode)
        else:
            return self._load_fallback(source_path, lead_time_hours, init_time, mode=current_mode)

    def _load_from_xarray(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime],
        mode: str = "DEMO"
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

        # Step / lead time selection
        lead_coord = next((c for c in ["step", "lead_time", "forecast_period", "time"] if c in ds.coords), None)
        step_idx = 0
        actual_lead_h = lead_time_hours

        if lead_coord and ds[lead_coord].size > 1:
            coord_vals = ds[lead_coord].values
            hours_list = []
            for val in coord_vals:
                if isinstance(val, np.timedelta64):
                    hours_list.append(float(val / np.timedelta64(1, "h")))
                elif hasattr(val, "total_seconds"):
                    hours_list.append(val.total_seconds() / 3600.0)
                else:
                    try:
                        hours_list.append(float(val))
                    except (ValueError, TypeError):
                        hours_list.append(0.0)
            hours_arr = np.array(hours_list)
            diffs = np.abs(hours_arr - lead_time_hours)
            min_diff = float(np.min(diffs))

            if min_diff > 0.5 and mode.upper() == "REAL":
                raise NWPValidationError(
                    f"LEAD_TIME_NOT_AVAILABLE: Requested lead +{lead_time_hours}h is not present in "
                    f"ECMWF forecast '{filepath}'. Available leads: {[int(h) for h in hours_arr]}. "
                    f"Silent selection of nearest lead is strictly prohibited in REAL mode."
                )

            step_idx = int(np.argmin(diffs))
            actual_lead_h = int(hours_arr[step_idx])

        var_mapping = {
            "rainfall_nwp": ["tp", "total_precipitation", "cp"],
            "temperature": ["2t", "t2m", "temperature"],
            "humidity": ["r", "rh", "humidity"],
            "pressure": ["msl", "sp", "mslp"],
            "cape": ["cape"],
            "vertical_velocity": ["w", "omega", "vvel500"],
            "geopotential_height": ["z", "gh", "z500"],
            "wind_u": ["10u", "u10"],
            "wind_v": ["10v", "v10"],
            "u_wind_850": ["u850", "u_850"],
            "v_wind_850": ["v850", "v_850"]
        }

        canonical_vars = {}
        var_attrs = {}

        for canonical, candidates in var_mapping.items():
            for cand in candidates:
                if cand in ds.variables:
                    da = ds[cand]
                    var_attrs[canonical] = dict(da.attrs)
                    data = da.values
                    if data.ndim == 3:
                        idx = min(step_idx, data.shape[0] - 1)
                        data = data[idx]
                    elif data.ndim == 4:
                        idx = min(step_idx, data.shape[0] - 1)
                        data = data[idx, 0]
                    canonical_vars[canonical] = data[np.ix_(lat_mask, lon_mask)]
                    break

        if "wind_u" in canonical_vars and "wind_v" in canonical_vars:
            u = canonical_vars["wind_u"]
            v = canonical_vars["wind_v"]
            canonical_vars["wind_speed"] = np.sqrt(u**2 + v**2)
            canonical_vars["wind_direction"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

        if "u_wind_850" in canonical_vars and "v_wind_850" in canonical_vars:
            canonical_vars["wind_speed_850"] = np.sqrt(canonical_vars["u_wind_850"]**2 + canonical_vars["v_wind_850"]**2)

        warnings = self.validate_variables(canonical_vars, var_attrs)

        now = init_time or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        valid = now + datetime.timedelta(hours=actual_lead_h)

        return {
            "provider": self.provider_name,
            "grid_lats": sub_lats,
            "grid_lons": sub_lons,
            "variables": canonical_vars,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "init_time": now.isoformat(),
                "valid_time": valid.isoformat(),
                "lead_time_hours": actual_lead_h,
                "requested_lead_time_hours": lead_time_hours,
                "step_index": step_idx,
                "warnings": warnings,
                "spatial_resolution": f"{self.resolution_deg}° x {self.resolution_deg}°",
                "variable_attributes": {k: v for k, v in var_attrs.items() if v}
            }
        }

    def _load_fallback(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime],
        mode: str = "DEMO"
    ) -> Dict[str, Any]:
        """Loads ECMWF tabular/JSON fallback or rejects if in REAL mode."""
        if mode.upper() == "REAL":
            raise NWPValidationError(
                f"ECMWF_AUTHENTIC_DATA_REQUIRED: Format for '{filepath}' cannot be substituted with generic GFS logic in REAL mode. "
                f"Authentic ECMWF HRES NetCDF or GRIB2 data is strictly required."
            )
        from src.data.nwp.gfs_adapter import GFSAdapter
        gfs = GFSAdapter()
        res = gfs.load_data(filepath, lead_time_hours, init_time, mode="DEMO")
        res["provider"] = self.provider_name
        return res


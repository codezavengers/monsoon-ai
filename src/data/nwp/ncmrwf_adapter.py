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
        init_time: Optional[datetime.datetime] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"NCMRWF data file not found at: {source_path}")

        current_mode = mode or os.environ.get("MODE", "DEMO")
        self.validate_data_mode_path(source_path, mode=current_mode)

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

            # Lead time coordinate selection
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

                if min_diff > 0.5 and current_mode.upper() == "REAL":
                    raise NWPValidationError(
                        f"LEAD_TIME_NOT_AVAILABLE: Requested lead +{lead_time_hours}h is not present in "
                        f"NCMRWF forecast '{source_path}'. Available leads: {[int(h) for h in hours_arr]}. "
                        f"Silent selection of nearest lead is strictly prohibited in REAL mode."
                    )

                step_idx = int(np.argmin(diffs))
                actual_lead_h = int(hours_arr[step_idx])

            var_mapping = {
                "rainfall_nwp": ["precip", "tot_prec", "precipitation_flux", "rain", "tp"],
                "temperature": ["temp", "t2m", "air_temperature"],
                "humidity": ["rh", "relative_humidity"],
                "pressure": ["mslp", "prmsl", "surface_air_pressure"],
                "cape": ["cape", "convective_available_potential_energy"],
                "vertical_velocity": ["vvel", "omega", "w"],
                "geopotential_height": ["gh", "z500"],
                "wind_u": ["u10", "u_wind"],
                "wind_v": ["v10", "v_wind"],
                "u_wind_850": ["u850", "u_wind_850"],
                "v_wind_850": ["v850", "v_wind_850"]
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
                    "source_file": os.path.basename(source_path),
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
        else:
            if current_mode.upper() == "REAL":
                raise NWPValidationError(
                    f"NCMRWF_AUTHENTIC_DATA_REQUIRED: Format for '{source_path}' cannot be substituted with generic GFS logic in REAL mode. "
                    f"Authentic NCMRWF NCUM NetCDF or GRIB2 data is strictly required."
                )
            from src.data.nwp.gfs_adapter import GFSAdapter
            gfs = GFSAdapter()
            res = gfs.load_data(source_path, lead_time_hours, init_time, mode="DEMO")
            res["provider"] = self.provider_name
            return res

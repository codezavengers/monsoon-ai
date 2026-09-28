"""
GFS (Global Forecast System - NOAA/NCEP) NWP Provider Adapter.
Handles GFS 0.25° gridded outputs in NetCDF, GRIB2 (via xarray/cfgrib), or normalized CSV/JSON.
Translates GFS GRIB parameters (APCP, PRMSL, TMP, RH, CAPE, VVEL, HGT) to canonical format.
"""

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np

from src.data.nwp.base import BaseNWPAdapter, NWPValidationError, INDIA_BBOX

# Optional scientific netcdf/grib libraries
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

class GFSAdapter(BaseNWPAdapter):
    """
    Adapter for NOAA GFS 0.25-degree Global Forecast System.
    GFS runs at 00Z, 06Z, 12Z, 18Z cycles with 3-hourly lead steps up to +120h.
    """
    def __init__(self):
        super().__init__(provider_name="GFS_0.25deg", resolution_deg=0.25)

    def load_data(
        self,
        source_path: str,
        lead_time_hours: int = 24,
        init_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Loads GFS forecast file. Supports:
        1. NetCDF (.nc, .nc4)
        2. GRIB2 (.grib2, .grb2) if cfgrib is present
        3. Structured meteorological array/dict JSON/CSV
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"GFS data file not found at: {source_path}")

        ext = os.path.splitext(source_path)[-1].lower()

        if ext in [".nc", ".nc4", ".grb2", ".grib2"] and XARRAY_AVAILABLE:
            return self._load_from_xarray(source_path, lead_time_hours, init_time)
        elif ext in [".json"]:
            return self._load_from_json(source_path, lead_time_hours, init_time)
        elif ext in [".csv"]:
            return self._load_from_csv(source_path, lead_time_hours, init_time)
        else:
            # Fallback parser for generic tabular/text format
            return self._load_from_csv(source_path, lead_time_hours, init_time)

    def _load_from_xarray(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime]
    ) -> Dict[str, Any]:
        """Reads NetCDF/GRIB2 via xarray, extracts exact lead time, and subsets to Indian domain."""
        ds = xr.open_dataset(filepath)

        # Standardize coordinate names
        lat_coord = next((c for c in ["lat", "latitude", "LAT"] if c in ds.coords), None)
        lon_coord = next((c for c in ["lon", "longitude", "LON"] if c in ds.coords), None)

        if not lat_coord or not lon_coord:
            raise NWPValidationError(f"Could not find lat/lon coordinates in {filepath}")

        lats = ds[lat_coord].values
        lons = self.normalize_longitudes(ds[lon_coord].values)

        # Subset to India Bounding Box with buffer
        lat_mask = (lats >= INDIA_BBOX["lat_min"] - 0.5) & (lats <= INDIA_BBOX["lat_max"] + 0.5)
        lon_mask = (lons >= INDIA_BBOX["lon_min"] - 0.5) & (lons <= INDIA_BBOX["lon_max"] + 0.5)

        sub_lats = lats[lat_mask]
        sub_lons = lons[lon_mask]
        self.validate_spatial_domain(sub_lats, sub_lons)

        # Lead time / Step index selection based on actual metadata
        lead_coord = next((c for c in ["step", "lead_time", "forecast_period", "lead", "time"] if c in ds.coords), None)
        step_idx = 0
        actual_lead_h = lead_time_hours

        if lead_coord and ds[lead_coord].size > 1:
            coord_vals = ds[lead_coord].values
            # Convert timedelta / hours to integer hours
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

            # Match exact or nearest lead time
            hours_arr = np.array(hours_list)
            diffs = np.abs(hours_arr - lead_time_hours)
            step_idx = int(np.argmin(diffs))
            actual_lead_h = int(hours_arr[step_idx])
        elif lead_coord and ds[lead_coord].size == 1:
            step_idx = 0

        # GFS variable translation mapping
        var_mapping = {
            "rainfall_nwp": ["apcp", "prate", "precip", "total_precipitation", "tp"],
            "temperature": ["tmp2m", "t2m", "tmp_2m", "temperature"],
            "humidity": ["rh2m", "r2", "rh_2m", "humidity"],
            "pressure": ["prmsl", "mslma", "pres_msl", "mslp"],
            "cape": ["cape", "capesfc"],
            "vertical_velocity": ["vvel500", "vvel", "omega500", "omega"],
            "geopotential_height": ["hgt500", "gh", "z500"],
            "wind_u": ["u10", "ugrd10m", "10u"],
            "wind_v": ["v10", "vgrd10m", "10v"],
            "u_wind_850": ["u850", "ugrd850mb", "u_850"],
            "v_wind_850": ["v850", "vgrd850mb", "v_850"],
            "humidity_850": ["rh850", "r850mb", "rh_850"]
        }

        canonical_vars = {}
        var_attrs = {}

        for canonical, candidates in var_mapping.items():
            for cand in candidates:
                if cand in ds.variables:
                    da = ds[cand]
                    var_attrs[canonical] = dict(da.attrs)
                    data = da.values

                    # Extract lead time slice if 3D/4D using validated step_idx
                    if data.ndim == 3:
                        idx = min(step_idx, data.shape[0] - 1)
                        data = data[idx]
                    elif data.ndim == 4:
                        idx = min(step_idx, data.shape[0] - 1)
                        data = data[idx, 0]

                    # Subset spatially
                    canonical_vars[canonical] = data[np.ix_(lat_mask, lon_mask)]
                    break

        # Compute 10m wind speed and direction if u and v are present
        if "wind_u" in canonical_vars and "wind_v" in canonical_vars:
            u = canonical_vars["wind_u"]
            v = canonical_vars["wind_v"]
            canonical_vars["wind_speed"] = np.sqrt(u**2 + v**2)
            canonical_vars["wind_direction"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

        # Compute 850hPa wind speed if 850hPa components are present
        if "u_wind_850" in canonical_vars and "v_wind_850" in canonical_vars:
            u850 = canonical_vars["u_wind_850"]
            v850 = canonical_vars["v_wind_850"]
            canonical_vars["wind_speed_850"] = np.sqrt(u850**2 + v850**2)

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

    def _load_from_json(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime]
    ) -> Dict[str, Any]:
        """Loads structured JSON forecast containing 2D or district grids."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        lats = np.array(data.get("lats", np.linspace(8.0, 36.0, 20)))
        lons = np.array(data.get("lons", np.linspace(70.0, 95.0, 20)))

        vars_dict = {}
        for k, v in data.get("variables", {}).items():
            vars_dict[k] = np.array(v, dtype=float)

        if "rainfall_nwp" not in vars_dict:
            vars_dict["rainfall_nwp"] = np.zeros((len(lats), len(lons)))

        self.validate_spatial_domain(lats, lons)
        warnings = self.validate_variables(vars_dict)

        now = init_time or datetime.datetime.utcnow()
        valid = now + datetime.timedelta(hours=lead_time_hours)

        return {
            "provider": self.provider_name,
            "grid_lats": lats,
            "grid_lons": lons,
            "variables": vars_dict,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "init_time": now.isoformat(),
                "valid_time": valid.isoformat(),
                "lead_time_hours": lead_time_hours,
                "warnings": warnings
            }
        }

    def _load_from_csv(
        self,
        filepath: str,
        lead_time_hours: int,
        init_time: Optional[datetime.datetime]
    ) -> Dict[str, Any]:
        """Loads tabular CSV containing lat, lon, and meteorological columns."""
        import csv
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if not rows:
            raise NWPValidationError(f"CSV file at {filepath} is empty.")

        lats = np.array(sorted(list(set(float(r["lat"]) for r in rows if "lat" in r))))
        lons = np.array(sorted(list(set(float(r["lon"]) for r in rows if "lon" in r))))

        if len(lats) == 0:
            lats = np.linspace(8.0, 36.0, 15)
            lons = np.linspace(70.0, 95.0, 15)

        grid_rain = np.zeros((len(lats), len(lons)))
        lat_idx = {round(lat, 2): i for i, lat in enumerate(lats)}
        lon_idx = {round(lon, 2): j for j, lon in enumerate(lons)}

        for r in rows:
            if "lat" in r and "lon" in r:
                la = round(float(r["lat"]), 2)
                lo = round(float(r["lon"]), 2)
                if la in lat_idx and lo in lon_idx:
                    grid_rain[lat_idx[la], lon_idx[lo]] = float(r.get("rainfall_nwp", 0.0))

        vars_dict = {
            "rainfall_nwp": grid_rain,
            "pressure": np.full((len(lats), len(lons)), 1004.0),
            "humidity": np.full((len(lats), len(lons)), 80.0),
            "temperature": np.full((len(lats), len(lons)), 28.0)
        }

        now = init_time or datetime.datetime.utcnow()
        valid = now + datetime.timedelta(hours=lead_time_hours)

        return {
            "provider": self.provider_name,
            "grid_lats": lats,
            "grid_lons": lons,
            "variables": vars_dict,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "init_time": now.isoformat(),
                "valid_time": valid.isoformat(),
                "lead_time_hours": lead_time_hours,
                "warnings": {}
            }
        }

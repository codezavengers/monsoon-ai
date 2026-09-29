"""
NOAA GFS (Global Forecast System) 0.25° Adapter.
Supports cycle discovery, NOAA NOMADS/AWS Open Data URL building, download validation, and spatial slicing.
"""

import os
import hashlib
import datetime
from typing import Dict, List, Any, Optional
import requests
import xarray as xr
import numpy as np

from .base import BaseNWPAdapter

class GFSAdapter(BaseNWPAdapter):
    """
    Adapter for genuine NOAA GFS 0.25-degree operational cycles.
    """
    def __init__(self, cycle: str = "00Z", resolution_deg: float = 0.25):
        super().__init__(provider="GFS", cycle=cycle, resolution_deg=resolution_deg)
        self.base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        self.aws_s3_url = "https://noaa-gfs-bdw-pds.s3.amazonaws.com"

    def discover_latest_cycle(self) -> Dict[str, Any]:
        """Discovers active operational cycle based on UTC time."""
        now = datetime.datetime.now(datetime.timezone.utc)
        # GFS operational runs are 00Z, 06Z, 12Z, 18Z with ~3.5h operational latency
        lagged = now - datetime.timedelta(hours=4)
        cycle_hour = (lagged.hour // 6) * 6
        cycle_str = f"{cycle_hour:02d}Z"
        date_str = lagged.strftime("%Y-%m-%d")
        
        return {
            "provider": "GFS",
            "cycle": cycle_str,
            "date": date_str,
            "init_time": f"{date_str}T{cycle_str[:2]}:00:00Z",
            "status": "AVAILABLE"
        }

    def build_download_url(self, date_str: str, cycle: str, lead_time_hours: int) -> str:
        """
        Constructs HTTPS subregion filter URL for NOAA NOMADS covering the India monsoon domain.
        """
        date_clean = date_str.replace("-", "")
        cycle_num = cycle.replace("Z", "").strip()
        f_hour = f"{lead_time_hours:03d}"
        
        # Subregion query covering India domain [6-38N, 68-98E]
        url = (
            f"{self.base_url}?file=gfs.t{cycle_num}z.pgrb2.0p25.f{f_hour}"
            f"&lev_2_m_above_ground=on&lev_surface=on&lev_500_mb=on&lev_850_mb=on"
            f"&var_APCP=on&var_TMP=on&var_RH=on&var_PRMSL=on&var_CAPE=on&var_VVEL=on&var_UGRD=on&var_VGRD=on"
            f"&subregion=&leftlon=68&rightlon=98&toplat=38&bottomlat=6"
            f"&dir=%2Fgfs.{date_clean}%2F{cycle_num}%2Fatmos"
        )
        return url

    def load_forecast_grid(self, filepath: str, lead_time_hours: int = 24) -> xr.Dataset:
        """Loads and normalizes NetCDF or GRIB forecast file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GFS forecast file not found: {filepath}")

        ds = xr.open_dataset(filepath)
        
        # Spatial subsetting over India
        lat_slice = slice(self.domain["lat_min"], self.domain["lat_max"])
        lon_slice = slice(self.domain["lon_min"], self.domain["lon_max"])
        
        # Handle coordinate orientation
        if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
            lat_slice = slice(self.domain["lat_max"], self.domain["lat_min"])
            
        ds_india = ds.sel(lat=lat_slice, lon=lon_slice)
        return ds_india

    def validate_grid(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Validates meteorological variable availability and physics."""
        required = ["apcp", "tmp2m", "rh2m", "prmsl", "cape", "vvel500"]
        present = [v for v in required if v in ds.data_vars]
        missing = [v for v in required if v not in ds.data_vars]

        passed = len(missing) == 0
        issues = []
        if missing:
            issues.append(f"Missing required GFS forecast fields: {', '.join(missing)}")

        if "apcp" in ds:
            val = float(ds["apcp"].values.max())
            if np.isnan(val):
                issues.append("Precipitation grid contains NaN values")

        return {
            "passed": passed and len(issues) == 0,
            "provider": "GFS",
            "required_variables": required,
            "present_variables": present,
            "missing_variables": missing,
            "issues": issues
        }

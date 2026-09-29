"""
NCMRWF (National Centre for Medium Range Weather Forecasting) NCUM Adapter.
Ingests India Unified Model (NCUM) regional and global operational forecasts.
"""

import os
import datetime
from typing import Dict, Any
import xarray as xr
from .base import BaseNWPAdapter

class NCMRWFAdapter(BaseNWPAdapter):
    """Adapter for NCMRWF NCUM forecasts."""
    def __init__(self, cycle: str = "00Z", resolution_deg: float = 0.25):
        super().__init__(provider="NCMRWF", cycle=cycle, resolution_deg=resolution_deg)
        self.portal_url = "https://www.ncmrwf.gov.in"

    def discover_latest_cycle(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        lagged = now - datetime.timedelta(hours=5)
        cycle_str = "00Z" if lagged.hour < 18 else "12Z"
        date_str = lagged.strftime("%Y-%m-%d")
        return {
            "provider": "NCMRWF",
            "cycle": cycle_str,
            "date": date_str,
            "init_time": f"{date_str}T{cycle_str[:2]}:00:00Z",
            "status": "AVAILABLE"
        }

    def build_download_url(self, date_str: str, cycle: str, lead_time_hours: int) -> str:
        date_clean = date_str.replace("-", "")
        return f"{self.portal_url}/ncum_fcst/{date_clean}_{cycle}_f{lead_time_hours:03d}.nc"

    def load_forecast_grid(self, filepath: str, lead_time_hours: int = 24) -> xr.Dataset:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NCMRWF file not found: {filepath}")
        return xr.open_dataset(filepath)

    def validate_grid(self, ds: xr.Dataset) -> Dict[str, Any]:
        has_rain = "rainfall" in ds.data_vars or "apcp" in ds.data_vars or "rain" in ds.data_vars
        return {
            "passed": has_rain,
            "provider": "NCMRWF",
            "issues": [] if has_rain else ["Precipitation field missing from NCMRWF grid"]
        }

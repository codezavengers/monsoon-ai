"""
ECMWF HRES (High Resolution) Adapter.
Handles cycle discovery, ECMWF Open Data format, and metadata alignment.
"""

import os
import datetime
from typing import Dict, Any
import xarray as xr
from .base import BaseNWPAdapter

class ECMWFAdapter(BaseNWPAdapter):
    """Adapter for ECMWF HRES model forecasts."""
    def __init__(self, cycle: str = "00Z", resolution_deg: float = 0.25):
        super().__init__(provider="ECMWF", cycle=cycle, resolution_deg=resolution_deg)
        self.base_url = "https://data.ecmwf.int/forecasts"

    def discover_latest_cycle(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        lagged = now - datetime.timedelta(hours=6)
        cycle_hour = (lagged.hour // 12) * 12
        cycle_str = f"{cycle_hour:02d}Z"
        date_str = lagged.strftime("%Y-%m-%d")
        return {
            "provider": "ECMWF",
            "cycle": cycle_str,
            "date": date_str,
            "init_time": f"{date_str}T{cycle_str[:2]}:00:00Z",
            "status": "AVAILABLE"
        }

    def build_download_url(self, date_str: str, cycle: str, lead_time_hours: int) -> str:
        date_clean = date_str.replace("-", "")
        cycle_num = cycle.replace("Z", "").strip()
        return f"{self.base_url}/{date_clean}/{cycle_num}z/ifs/0p25/oper/{date_clean}{cycle_num}0000-{lead_time_hours}h-oper-fc.grib2"

    def load_forecast_grid(self, filepath: str, lead_time_hours: int = 24) -> xr.Dataset:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ECMWF file not found: {filepath}")
        return xr.open_dataset(filepath)

    def validate_grid(self, ds: xr.Dataset) -> Dict[str, Any]:
        has_tp = "tp" in ds.data_vars or "apcp" in ds.data_vars
        return {
            "passed": has_tp,
            "provider": "ECMWF",
            "issues": [] if has_tp else ["Total precipitation field 'tp' missing from ECMWF grid"]
        }

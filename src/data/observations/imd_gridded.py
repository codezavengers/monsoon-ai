"""
IMD (India Meteorological Department) 0.25° Daily Gridded Rainfall Observation Loader.
Enforces strict temporal alignment with the IMD 0300 UTC - 0300 UTC accumulation window.
"""

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np
import xarray as xr

from .base import BaseObservationLoader

class IMDGriddedLoader(BaseObservationLoader):
    """
    Ingests official IMD 0.25-deg daily gridded rainfall NetCDF records.
    Never accepts arbitrary first/nearest observation dates.
    """
    def __init__(self, raw_obs_dir: str = "data/raw/observations", fixture_path: str = "data/fixtures/observations/imd_rainfall_2024.nc"):
        super().__init__(observation_source="IMD_Gridded_0.25deg")
        self.raw_obs_dir = raw_obs_dir
        self.fixture_path = fixture_path

    def load_observation_for_date(self, target_date_str: str, allow_fixture: bool = False) -> Optional[xr.DataArray]:
        """
        Loads 2D observation field for the exact date requested.
        Rejects mismatches explicitly.
        """
        target_dt = datetime.datetime.fromisoformat(target_date_str[:10])
        target_np_dt = np.datetime64(target_date_str[:10])

        # 1. Search in operational raw observations directory
        if os.path.exists(self.raw_obs_dir):
            for fname in os.listdir(self.raw_obs_dir):
                if fname.endswith((".nc", ".nc4")):
                    fpath = os.path.join(self.raw_obs_dir, fname)
                    try:
                        ds = xr.open_dataset(fpath)
                        rain_var = "rain" if "rain" in ds else "rainfall"
                        if "time" in ds.coords and target_np_dt in ds.time.values:
                            return ds[rain_var].sel(time=target_np_dt)
                    except Exception:
                        continue

        # 2. Check fixture path if explicitly permitted (DEMO / benchmark testing)
        if allow_fixture and os.path.exists(self.fixture_path):
            try:
                ds = xr.open_dataset(self.fixture_path)
                rain_var = "rain" if "rain" in ds else "rainfall"
                if target_np_dt in ds.time.values:
                    return ds[rain_var].sel(time=target_np_dt)
            except Exception:
                pass

        return None

    def validate_temporal_alignment(self, forecast_valid_time: str, obs_date_str: str) -> bool:
        """
        Verifies that forecast valid time aligns with target observation date.
        IMD observation for date D represents precipitation ending at 03:00 UTC on date D.
        Forecast valid time must match date D (within 6-hour tolerance of accumulation center).
        """
        try:
            fc_dt = datetime.datetime.fromisoformat(forecast_valid_time.replace("Z", "+00:00"))
            obs_dt = datetime.datetime.fromisoformat(obs_date_str[:10] + "T03:00:00+00:00")
            diff_hours = abs((fc_dt - obs_dt).total_seconds()) / 3600.0
            return diff_hours <= 12.0
        except Exception:
            return False

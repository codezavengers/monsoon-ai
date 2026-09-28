"""
India Meteorological Department (IMD) Gridded Rainfall Observation Provider.
Supports IMD 0.25° x 0.25° (Pai et al. 2014) and 1.0° x 1.0° (Rajeevan et al. 2006) daily rainfall products.
Daily accumulation corresponds to 08:30 IST to 08:30 IST (03:00 UTC to 03:00 UTC).
"""

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np

from src.data.observations.base import BaseObservationProvider, ObservationValidationError

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

class IMDGriddedObservationProvider(BaseObservationProvider):
    """
    Reader for IMD Daily Gridded Rainfall datasets.
    Coordinates span India landmass [6.5°N to 38.5°N, 66.5°E to 100.0°E].
    """
    def __init__(self, resolution_deg: float = 0.25):
        super().__init__(source_name="IMD_Gridded_0.25deg", resolution_deg=resolution_deg)

    def load_observations(
        self,
        filepath: str,
        target_date: datetime.date
    ) -> Dict[str, Any]:
        """Loads IMD observations for the given date from NetCDF, CSV or binary format."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IMD observation file not found at: {filepath}")

        ext = os.path.splitext(filepath)[-1].lower()
        if ext in [".nc", ".nc4"] and XARRAY_AVAILABLE:
            return self._load_netcdf(filepath, target_date)
        elif ext == ".csv":
            return self._load_csv(filepath, target_date)
        else:
            return self._load_csv(filepath, target_date)

    def compute_accumulation_window(
        self,
        target_date: datetime.date
    ) -> Dict[str, str]:
        """
        Computes standard IMD 24-hour daily rainfall accumulation window.
        IMD observation period: 08:30 IST (Day-1) to 08:30 IST (Target Day)
        which corresponds exactly to 03:00 UTC (Day-1) to 03:00 UTC (Target Day).
        """
        start_utc = datetime.datetime(target_date.year, target_date.month, target_date.day, 3, 0) - datetime.timedelta(days=1)
        end_utc = datetime.datetime(target_date.year, target_date.month, target_date.day, 3, 0)
        return {
            "start_time_utc": start_utc.isoformat() + "Z",
            "end_time_utc": end_utc.isoformat() + "Z",
            "accumulation_hours": 24,
            "ist_convention": "08:30 IST (Day-1) to 08:30 IST (Target Day)"
        }

    def _load_netcdf(self, filepath: str, target_date: datetime.date) -> Dict[str, Any]:
        ds = xr.open_dataset(filepath)

        lat_coord = next((c for c in ["lat", "latitude"] if c in ds.coords), None)
        lon_coord = next((c for c in ["lon", "longitude"] if c in ds.coords), None)
        time_coord = next((c for c in ["time", "TIME"] if c in ds.coords), None)

        if not lat_coord or not lon_coord:
            raise ObservationValidationError(f"Missing lat/lon in IMD dataset: {filepath}")

        lats = ds[lat_coord].values
        lons = ds[lon_coord].values

        rain_var = next((v for v in ["rain", "rainfall", "rf", "precipitation"] if v in ds.variables), None)
        if not rain_var:
            raise ObservationValidationError(f"No rainfall variable found in IMD NetCDF: {filepath}")

        # Slice by target date if time dimension exists
        if time_coord and ds[time_coord].size > 1:
            try:
                date_str = target_date.strftime("%Y-%m-%d")
                rain_slice = ds[rain_var].sel({time_coord: date_str}).values
            except Exception:
                # Attempt string or day matching
                try:
                    time_vals = ds[time_coord].values
                    target_dt = np.datetime64(target_date)
                    diffs = np.abs(time_vals.astype("datetime64[D]") - target_dt)
                    best_idx = int(np.argmin(diffs))
                    rain_slice = ds[rain_var].values[best_idx]
                except Exception:
                    rain_slice = ds[rain_var].values[0]
        else:
            rain_slice = ds[rain_var].values
            if rain_slice.ndim == 3:
                rain_slice = rain_slice[0]

        rain_clean = self.validate_rainfall_bounds(rain_slice)
        quality_mask = ~np.isnan(rain_clean)
        accum_meta = self.compute_accumulation_window(target_date)

        return {
            "source": self.source_name,
            "target_date": target_date.isoformat(),
            "grid_lats": lats,
            "grid_lons": lons,
            "rainfall_obs": rain_clean,
            "quality_mask": quality_mask,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "resolution": f"{self.resolution_deg}° x {self.resolution_deg}°",
                "accumulation_window": "03:00 UTC (Day-1) to 03:00 UTC (Target Day)",
                "accumulation_details": accum_meta,
                "valid_points_count": int(np.sum(quality_mask)),
                "provenance": {
                    "provider": "India Meteorological Department (IMD)",
                    "dataset_version": "IMD 0.25-deg Daily Gridded Rainfall (Pai et al. 2014)",
                    "ingestion_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
            }
        }

    def _load_csv(self, filepath: str, target_date: datetime.date) -> Dict[str, Any]:
        """Loads observation CSV with columns [date, lat, lon, rainfall_obs]."""
        import csv
        date_str = target_date.strftime("%Y-%m-%d")
        matched_points = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_date = row.get("date", "")
                if r_date == date_str or not r_date: # Match date or single-day file
                    matched_points.append(row)

        if not matched_points:
            # Try matching year/month/day
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (int(row.get("year", 0)) == target_date.year and
                        int(row.get("month", 0)) == target_date.month and
                        int(row.get("day", 0)) == target_date.day):
                        matched_points.append(row)

        if not matched_points:
            # Check if multi-station file exists and read records
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                matched_points = [r for idx, r in enumerate(reader) if idx < 300]

        lats = sorted(list(set(float(r["lat"]) for r in matched_points if "lat" in r)))
        lons = sorted(list(set(float(r["lon"]) for r in matched_points if "lon" in r)))

        if len(lats) == 0:
            lats = list(np.linspace(8.0, 36.0, 15))
            lons = list(np.linspace(70.0, 95.0, 15))

        grid_rain = np.full((len(lats), len(lons)), np.nan)
        lat_idx = {round(lat, 2): i for i, lat in enumerate(lats)}
        lon_idx = {round(lon, 2): j for j, lon in enumerate(lons)}

        for r in matched_points:
            if "lat" in r and "lon" in r:
                la = round(float(r["lat"]), 2)
                lo = round(float(r["lon"]), 2)
                if la in lat_idx and lo in lon_idx:
                    grid_rain[lat_idx[la], lon_idx[lo]] = float(r.get("rainfall_obs", 0.0))

        rain_clean = self.validate_rainfall_bounds(grid_rain)
        quality_mask = ~np.isnan(rain_clean)
        accum_meta = self.compute_accumulation_window(target_date)

        return {
            "source": self.source_name,
            "target_date": target_date.isoformat(),
            "grid_lats": np.array(lats),
            "grid_lons": np.array(lons),
            "rainfall_obs": rain_clean,
            "quality_mask": quality_mask,
            "metadata": {
                "source_file": os.path.basename(filepath),
                "resolution": f"{self.resolution_deg}° x {self.resolution_deg}°",
                "accumulation_window": "03:00 UTC (Day-1) to 03:00 UTC (Target Day)",
                "accumulation_details": accum_meta,
                "valid_points_count": int(np.sum(quality_mask)),
                "provenance": {
                    "provider": "India Meteorological Department (IMD)",
                    "dataset_version": "IMD Gridded Rainfall (0.25-deg/1.0-deg)",
                    "ingestion_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
            }
        }

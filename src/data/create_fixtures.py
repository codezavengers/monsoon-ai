"""
Creates or validates NetCDF and CSV fixtures for testing and reproducible benchmarking.
"""

import os
import datetime
import numpy as np
import xarray as xr

def create_fixtures(fixtures_dir: str = "data/fixtures"):
    """Generates synthetic NWP and IMD NetCDF fixtures if not already present."""
    os.makedirs(os.path.join(fixtures_dir, "nwp"), exist_ok=True)
    os.makedirs(os.path.join(fixtures_dir, "observations"), exist_ok=True)

    nwp_file = os.path.join(fixtures_dir, "nwp", "gfs_20240715_00z.nc")
    obs_file = os.path.join(fixtures_dir, "observations", "imd_rainfall_2024.nc")

    lats = np.arange(8.0, 36.5, 0.5)
    lons = np.arange(68.5, 96.5, 0.5)
    steps = [
        np.timedelta64(6, 'h'), np.timedelta64(12, 'h'), np.timedelta64(24, 'h'),
        np.timedelta64(48, 'h'), np.timedelta64(72, 'h'), np.timedelta64(120, 'h')
    ]

    if not os.path.exists(nwp_file):
        n_steps = len(steps)
        n_lat = len(lats)
        n_lon = len(lons)
        
        apcp = np.random.gamma(shape=1.5, scale=12.0, size=(n_steps, n_lat, n_lon)).astype(np.float32)
        tmp2m = np.random.normal(loc=27.0, scale=3.0, size=(n_steps, n_lat, n_lon)).astype(np.float32)
        rh2m = np.clip(np.random.normal(loc=80.0, scale=10.0, size=(n_steps, n_lat, n_lon)), 20.0, 100.0).astype(np.float32)
        prmsl = np.random.normal(loc=1002.0, scale=5.0, size=(n_steps, n_lat, n_lon)).astype(np.float32)
        cape = np.random.exponential(scale=1600.0, size=(n_steps, n_lat, n_lon)).astype(np.float32)
        vvel500 = np.random.normal(loc=-0.2, scale=0.3, size=(n_steps, n_lat, n_lon)).astype(np.float32)

        ds_nwp = xr.Dataset(
            data_vars={
                "apcp": (("step", "lat", "lon"), apcp),
                "tmp2m": (("step", "lat", "lon"), tmp2m),
                "rh2m": (("step", "lat", "lon"), rh2m),
                "prmsl": (("step", "lat", "lon"), prmsl),
                "cape": (("step", "lat", "lon"), cape),
                "vvel500": (("step", "lat", "lon"), vvel500),
            },
            coords={"step": steps, "lat": lats, "lon": lons},
            attrs={
                "title": "NOAA GFS 0.25-deg Operational Forecast Fixture",
                "institution": "NCEP/NOAA",
                "reference_time": "2024-07-15T00:00:00Z"
            }
        )
        ds_nwp.to_netcdf(nwp_file)

    if not os.path.exists(obs_file):
        dates = [np.datetime64("2024-07-15"), np.datetime64("2024-08-03"), np.datetime64("2024-08-20")]
        rain = np.random.gamma(shape=1.4, scale=14.0, size=(len(dates), len(lats), len(lons))).astype(np.float32)
        ds_obs = xr.Dataset(
            data_vars={"rain": (("time", "lat", "lon"), rain)},
            coords={"time": dates, "lat": lats, "lon": lons},
            attrs={
                "title": "IMD 0.25-deg Daily Gridded Rainfall Observation Fixture",
                "institution": "India Meteorological Department (IMD)",
                "accumulation_window": "03:00 UTC (Day-1) to 03:00 UTC (Target Day)"
            }
        )
        ds_obs.to_netcdf(obs_file)

if __name__ == "__main__":
    create_fixtures()

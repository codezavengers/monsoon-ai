"""
NWP Provider and Adapter Architecture for Indian Monsoon Precipitation Forecasting.
Standardizes NWP inputs from GFS (NCEP), ECMWF, and NCMRWF (MoES) into canonical units and coordinates.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import datetime
import numpy as np
import os
import json

# Canonical NWP variable names required for Indian Monsoon regime-aware post-processing
CANONICAL_VARIABLES = {
    "rainfall_nwp": "NWP Accumulated Surface Rainfall (mm/day)",
    "temperature": "2m Air Temperature (°C)",
    "humidity": "2m Relative Humidity (%)",
    "pressure": "Mean Sea Level Pressure (hPa)",
    "wind_speed": "10m Wind Speed (m/s)",
    "wind_direction": "10m Wind Direction (degrees 0-360)",
    "cape": "Surface Convective Available Potential Energy (J/kg)",
    "vertical_velocity": "500 hPa Vertical Velocity / Omega (Pa/s)",
    "geopotential_height": "500 hPa Geopotential Height (gpm)",
    "u_wind_850": "850 hPa Zonal Wind / Low-Level Jet (m/s)",
    "v_wind_850": "850 hPa Meridional Wind (m/s)"
}

# Indian Subcontinent geographic bounding box
INDIA_BBOX = {
    "lat_min": 6.0,
    "lat_max": 38.0,
    "lon_min": 68.0,
    "lon_max": 98.0
}

class NWPValidationError(Exception):
    """Raised when NWP data fails unit, coordinate, or variable completeness checks."""
    pass

class BaseNWPAdapter(ABC):
    """
    Abstract Base Class for NWP Data Providers.
    Normalizes variable names, units, spatial coordinates, and temporal metadata.
    """
    def __init__(self, provider_name: str, resolution_deg: float = 0.25):
        self.provider_name = provider_name
        self.resolution_deg = resolution_deg
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def load_data(
        self, 
        source_path: str, 
        lead_time_hours: int = 24,
        init_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Loads and standardizes NWP data for a specified lead time.
        
        Returns a dict containing:
          - 'grid_lats': 1D array of latitudes
          - 'grid_lons': 1D array of longitudes
          - 'variables': dict of 2D numpy arrays [lat x lon] with canonical variable names
          - 'metadata': dict with initialization time, valid time, lead time, and provider
        """
        pass

    def validate_spatial_domain(self, lats: np.ndarray, lons: np.ndarray) -> bool:
        """Validates that coordinates span the Indian subcontinent."""
        if len(lats) == 0 or len(lons) == 0:
            raise NWPValidationError(f"Empty coordinate arrays in {self.provider_name} NWP data.")
            
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
        
        # Check intersection with Indian subcontinent
        if lat_max < INDIA_BBOX["lat_min"] or lat_min > INDIA_BBOX["lat_max"]:
            raise NWPValidationError(f"Latitudes [{lat_min}, {lat_max}] do not cover Indian domain [6, 38]°N")
        if lon_max < INDIA_BBOX["lon_min"] or lon_min > INDIA_BBOX["lon_max"]:
            raise NWPValidationError(f"Longitudes [{lon_min}, {lon_max}] do not cover Indian domain [68, 98]°E")
            
        return True

    def validate_variables(self, variables: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Validates canonical variables, bounds, and flags missing predictors."""
        warnings = {}
        
        # Core required variable: rainfall_nwp
        if "rainfall_nwp" not in variables:
            raise NWPValidationError("Missing critical variable: 'rainfall_nwp'")
            
        rain = variables["rainfall_nwp"]
        if np.any(np.isnan(rain)):
            warnings["rainfall_nan"] = f"Rainfall field contains {np.sum(np.isnan(rain))} NaN grid cells (interpolated/zeroed)."
        if np.any(rain < 0):
            warnings["rainfall_negative"] = "Negative rainfall detected in NWP data, clipped to 0.0 mm."
            variables["rainfall_nwp"] = np.clip(rain, 0.0, None)
            
        # Physical bounds checks
        if "pressure" in variables:
            p = variables["pressure"]
            # Auto-convert Pa to hPa if mean pressure > 50000 Pa
            if np.nanmean(p) > 20000.0:
                variables["pressure"] = p / 100.0
                
        if "temperature" in variables:
            t = variables["temperature"]
            # Auto-convert Kelvin to Celsius
            if np.nanmean(t) > 200.0:
                variables["temperature"] = t - 273.15
                
        if "humidity" in variables:
            rh = variables["humidity"]
            # Auto-convert 0-1 fraction to percentage
            if np.nanmax(rh) <= 1.01:
                variables["humidity"] = rh * 100.0
                
        return warnings

    @staticmethod
    def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
        """Converts longitudes from [0, 360) format to standard [-180, 180) format if needed."""
        lons_norm = np.copy(lons)
        lons_norm[lons_norm > 180.0] -= 360.0
        return lons_norm

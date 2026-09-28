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
    "wind_u": "10m Zonal Wind Component (m/s)",
    "wind_v": "10m Meridional Wind Component (m/s)",
    "u_wind_850": "850 hPa Zonal Wind / Low-Level Jet (m/s)",
    "v_wind_850": "850 hPa Meridional Wind (m/s)",
    "wind_speed_850": "850 hPa Wind Speed Magnitude (m/s)",
    "humidity_850": "850 hPa Relative Humidity (%)",
    "cape": "Surface Convective Available Potential Energy (J/kg)",
    "vertical_velocity": "500 hPa Vertical Velocity / Omega (Pa/s)",
    "geopotential_height": "500 hPa Geopotential Height (gpm)"
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
    Uses metadata attributes when available, and issues explicit warnings on fallbacks.
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

    def normalize_variable_with_metadata(
        self, 
        canonical_name: str, 
        data: np.ndarray, 
        attrs: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Normalizes NWP variables using dataset metadata / attributes (units, long_name).
        Returns normalized array and an optional warning string.
        """
        data = np.asarray(data, dtype=float).copy()
        attrs = attrs or {}
        raw_units = str(attrs.get("units", "") or attrs.get("unit", "")).strip().lower()
        warning = None

        if canonical_name == "rainfall_nwp":
            # Precipitation conversion: mm, m, kg m-2 s-1, or m/s
            if raw_units in ["m", "meter", "meters"]:
                data = data * 1000.0
            elif raw_units in ["kg m-2 s-1", "kg/m2/s", "kg m**-2 s**-1", "mm/s"]:
                # Convert precipitation flux (kg/m2/s = mm/s) to 24-hr accumulation
                data = data * 86400.0
            elif raw_units in ["mm", "mm/day", "kg m-2", "kg/m2"]:
                pass # Already mm
            else:
                # Value heuristic fallback with explicit warning
                max_val = float(np.nanmax(data)) if np.any(~np.isnan(data)) else 0.0
                if max_val < 2.0 and max_val > 0.0:
                    warning = f"Unknown rainfall units '{raw_units}'; converted from meters to mm based on max value {max_val:.4f}."
                    data = data * 1000.0
                elif not raw_units:
                    warning = "Rainfall variable has no units metadata; assuming mm/day."
            # Ensure physical lower bound
            data = np.clip(data, 0.0, None)

        elif canonical_name == "temperature":
            # Temperature: Kelvin to Celsius
            if raw_units in ["k", "kelvin", "degk", "degree_k"]:
                data = data - 273.15
            elif raw_units in ["c", "celsius", "degc", "degree_c"]:
                pass
            else:
                mean_t = float(np.nanmean(data)) if np.any(~np.isnan(data)) else 0.0
                if mean_t > 150.0:
                    data = data - 273.15
                    warning = f"Unknown temperature units '{raw_units}'; converted from Kelvin to Celsius (mean={mean_t:.1f}K)."

        elif canonical_name in ["humidity", "humidity_850"]:
            # Humidity: fraction (0-1) to percentage (0-100)
            if raw_units in ["fraction", "1", "0-1", "ratio"]:
                data = data * 100.0
            elif raw_units in ["%", "percent", "percentage"]:
                pass
            else:
                max_h = float(np.nanmax(data)) if np.any(~np.isnan(data)) else 0.0
                if max_h <= 1.01 and max_h > 0.0:
                    data = data * 100.0
                    warning = f"Unknown humidity units '{raw_units}'; converted fraction [0, 1] to percentage (max={max_h:.2f})."
            data = np.clip(data, 0.0, 100.0)

        elif canonical_name == "pressure":
            # Pressure: Pa to hPa
            if raw_units in ["pa", "pascal", "pascals"]:
                data = data / 100.0
            elif raw_units in ["hpa", "hectopascal", "mbar", "millibar"]:
                pass
            else:
                mean_p = float(np.nanmean(data)) if np.any(~np.isnan(data)) else 0.0
                if mean_p > 20000.0:
                    data = data / 100.0
                    warning = f"Unknown pressure units '{raw_units}'; converted Pa to hPa (mean={mean_p:.0f} Pa)."

        return data, warning

    def validate_variables(
        self, 
        variables: Dict[str, np.ndarray],
        var_attrs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Validates canonical variables, bounds, and normalizes using metadata."""
        warnings = {}
        var_attrs = var_attrs or {}
        
        # Core required variable: rainfall_nwp
        if "rainfall_nwp" not in variables:
            raise NWPValidationError("Missing critical variable: 'rainfall_nwp'")
            
        for var_name in list(variables.keys()):
            attrs = var_attrs.get(var_name, {})
            norm_data, warn = self.normalize_variable_with_metadata(var_name, variables[var_name], attrs)
            variables[var_name] = norm_data
            if warn:
                warnings[f"{var_name}_units"] = warn

        # Check for NaN / negative rainfall
        rain = variables["rainfall_nwp"]
        if np.any(np.isnan(rain)):
            warnings["rainfall_nan"] = f"Rainfall field contains {int(np.sum(np.isnan(rain)))} NaN grid cells (zeroed)."
            variables["rainfall_nwp"] = np.nan_to_num(rain, nan=0.0)
        if np.any(rain < 0):
            warnings["rainfall_negative"] = "Negative rainfall detected in NWP data, clipped to 0.0 mm."
            variables["rainfall_nwp"] = np.clip(rain, 0.0, None)

        # Derive 10m wind speed if wind_u and wind_v are present
        if "wind_u" in variables and "wind_v" in variables and "wind_speed" not in variables:
            u10 = variables["wind_u"]
            v10 = variables["wind_v"]
            variables["wind_speed"] = np.sqrt(u10**2 + v10**2)
            variables["wind_direction"] = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0

        # Derive 850 hPa wind speed ONLY if 850 hPa components are explicitly present
        if "u_wind_850" in variables and "v_wind_850" in variables and "wind_speed_850" not in variables:
            u850 = variables["u_wind_850"]
            v850 = variables["v_wind_850"]
            variables["wind_speed_850"] = np.sqrt(u850**2 + v850**2)

        return warnings

    @staticmethod
    def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
        """Converts longitudes from [0, 360) format to standard [-180, 180) format if needed."""
        lons_norm = np.copy(lons)
        lons_norm[lons_norm > 180.0] -= 360.0
        return lons_norm

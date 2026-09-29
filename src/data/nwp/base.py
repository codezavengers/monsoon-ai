"""
Abstract Base Class for Numerical Weather Prediction (NWP) model adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import datetime
import xarray as xr

class BaseNWPAdapter(ABC):
    """
    Standard interface for operational NWP forecast acquisition, ingestion, and validation.
    """
    def __init__(self, provider: str, cycle: str = "00Z", resolution_deg: float = 0.25):
        self.provider = provider
        self.cycle = cycle
        self.resolution_deg = resolution_deg
        # India meteorological domain [lat_min, lat_max, lon_min, lon_max]
        self.domain = {"lat_min": 6.0, "lat_max": 38.0, "lon_min": 68.0, "lon_max": 98.0}

    @abstractmethod
    def discover_latest_cycle(self) -> Dict[str, Any]:
        """Discovers latest available initialization cycle from remote provider."""
        pass

    @abstractmethod
    def build_download_url(self, date_str: str, cycle: str, lead_time_hours: int) -> str:
        """Constructs remote HTTP/HTTPS download URL for given cycle and lead time."""
        pass

    @abstractmethod
    def load_forecast_grid(self, filepath: str, lead_time_hours: int) -> xr.Dataset:
        """Loads and spatially slices 2D/3D meteorological variables over India."""
        pass

    @abstractmethod
    def validate_grid(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Validates variables, units, bounding box, and physical range."""
        pass

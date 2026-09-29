"""
Base Observation Loader for Meteorological Ground Truth Data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import datetime
import xarray as xr

class BaseObservationLoader(ABC):
    """Abstract interface for rainfall observation dataset ingestion."""
    def __init__(self, observation_source: str = "IMD_Gridded_0.25deg"):
        self.source = observation_source
        self.accumulation_window = "03:00 UTC (Day-1) to 03:00 UTC (Target Day)"

    @abstractmethod
    def load_observation_for_date(self, target_date_str: str) -> Optional[xr.DataArray]:
        """Loads 24-hr cumulative ground-truth rainfall for specific target date."""
        pass

    @abstractmethod
    def validate_temporal_alignment(self, forecast_valid_time: str, obs_date_str: str) -> bool:
        """Validates exact temporal synchronization between NWP valid window and observation window."""
        pass

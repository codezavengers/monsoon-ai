"""
Independent Rainfall Observation Ingestion Pipeline.
Supports real observational truth products (IMD daily gridded rainfall, automatic weather stations).
Strictly decoupled from NWP generation in REAL mode.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import datetime
import numpy as np
import os

class ObservationValidationError(Exception):
    """Raised when observation data is invalid, unaligned, or missing."""
    pass

class BaseObservationProvider(ABC):
    """
    Abstract Base Class for Observational Precipitation Truth Data.
    Tracks provenance, observation accumulation window (e.g. 03:00 to 03:00 UTC),
    and quality flags.
    """
    def __init__(self, source_name: str, resolution_deg: float = 0.25):
        self.source_name = source_name
        self.resolution_deg = resolution_deg
        
    @abstractmethod
    def load_observations(
        self,
        filepath: str,
        target_date: datetime.date
    ) -> Dict[str, Any]:
        """
        Loads ground-truth observations for the specified target date.
        
        Returns dict with:
          - 'grid_lats': 1D latitude array
          - 'grid_lons': 1D longitude array
          - 'rainfall_obs': 2D numpy array [lat x lon] in mm/day
          - 'quality_mask': 2D boolean array (True if valid observation)
          - 'metadata': dict with source, accumulation window, and provenance
        """
        pass

    def validate_rainfall_bounds(self, rain: np.ndarray) -> np.ndarray:
        """Physical bounds validation: rain >= 0.0, no NaNs on valid land."""
        clean = np.copy(rain)
        # Extreme world record 24h rain is ~1825mm; in India Cherrapunji record is ~1040mm/day
        clean[clean < 0.0] = 0.0
        clean[clean > 2500.0] = np.nan
        return clean

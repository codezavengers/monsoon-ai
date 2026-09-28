"""
Data validation module for meteorological variables and NWP forecasts.
Ensures physical plausibility, coordinate ranges, and absence of NaNs/Infs.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

# Physical bounds for meteorological variables in India
METEOROLOGICAL_BOUNDS = {
    "lat": (6.0, 38.0),             # India latitude range
    "lon": (68.0, 98.0),            # India longitude range
    "rainfall_nwp": (0.0, 1500.0),  # Max daily rain record ~1000mm+ in Cherrapunji
    "rainfall_obs": (0.0, 1500.0),
    "temperature": (-15.0, 55.0),   # °C
    "humidity": (0.0, 100.0),       # %
    "pressure": (850.0, 1050.0),    # hPa
    "wind_speed": (0.0, 80.0),      # m/s
    "wind_direction": (0.0, 360.0), # degrees
    "cape": (0.0, 8000.0),          # J/kg
    "elevation": (-10.0, 9000.0),   # meters
    "coast_dist_km": (0.0, 2000.0)  # km
}

def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a single meteorological record against physical bounds and non-null constraints.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    for key, (low, high) in METEOROLOGICAL_BOUNDS.items():
        if key in record and record[key] is not None:
            val = record[key]
            if np.isnan(val) or np.isinf(val):
                errors.append(f"{key} is NaN or Inf")
            elif val < low or val > high:
                errors.append(f"{key}={val} out of physical bounds [{low}, {high}]")
                
    if record.get("rainfall_nwp", 0) < 0:
        errors.append("Negative NWP rainfall not permitted")
        
    return len(errors) == 0, errors

def clean_and_validate_dataset(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Cleans dataset by imputing or clipping minor out-of-bound variables and filtering corrupted records.
    """
    cleaned = []
    stats = {"total": len(records), "valid": 0, "fixed": 0, "dropped": 0}
    
    for r in records:
        row = dict(r)
        # Fix negative rainfall
        if "rainfall_nwp" in row and (row["rainfall_nwp"] is None or row["rainfall_nwp"] < 0 or np.isnan(row["rainfall_nwp"])):
            row["rainfall_nwp"] = 0.0
            stats["fixed"] += 1
            
        if "rainfall_obs" in row and (row["rainfall_obs"] is None or row["rainfall_obs"] < 0 or np.isnan(row["rainfall_obs"])):
            row["rainfall_obs"] = 0.0
            stats["fixed"] += 1
            
        # Clip humidity to [0, 100]
        if "humidity" in row and row["humidity"] is not None:
            row["humidity"] = max(0.0, min(100.0, float(row["humidity"])))
            
        is_valid, _ = validate_record(row)
        if is_valid:
            cleaned.append(row)
            stats["valid"] += 1
        else:
            stats["dropped"] += 1
            
    return cleaned, stats

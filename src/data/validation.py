"""
Data validation and physical consistency auditing.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def validate_meteorological_record(record: Dict[str, Any], mode: str = "DEMO") -> Tuple[bool, List[str]]:
    """Validates physical atmospheric boundaries and completeness."""
    issues = []
    
    # Required core variables
    required = ["rainfall", "humidity", "temperature", "pressure"]
    if mode == "REAL":
        for r in required:
            if r not in record and f"{r}_nwp" not in record:
                issues.append(f"Missing required field in REAL mode: {r}")

    # Bounds check
    rain = record.get("rainfall_nwp", record.get("rainfall", 0.0))
    if rain is not None and float(rain) < 0.0:
        issues.append(f"Negative rainfall: {rain} mm")

    rh = record.get("humidity", 50.0)
    if rh is not None and (float(rh) < 0.0 or float(rh) > 100.0):
        issues.append(f"Relative humidity out of bounds [0, 100]: {rh}%")

    pres = record.get("pressure", 1000.0)
    if pres is not None and (float(pres) < 850.0 or float(pres) > 1050.0):
        issues.append(f"Barometric pressure out of physical range [850, 1050]: {pres} hPa")

    return (len(issues) == 0, issues)

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Audits entire DataFrame for missing rows, NaN rates, and target alignment."""
    issues = []
    if len(df) == 0:
        return (False, ["Dataset is empty"])
    
    null_counts = df.isnull().sum()
    for col, n in null_counts.items():
        if n > 0:
            issues.append(f"Column '{col}' has {n} null values ({n/len(df):.1%})")
            
    if "rainfall_nwp" in df.columns:
        neg = (df["rainfall_nwp"] < 0).sum()
        if neg > 0:
            issues.append(f"rainfall_nwp contains {neg} negative values")
            
    return (len(issues) == 0, issues)

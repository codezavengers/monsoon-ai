"""
Physical constraints and transformations for precipitation post-processing.
Enforces non-negativity and handles skewed distributions using log1p/expm1.
"""

from typing import Union
import numpy as np

def apply_physical_constraints(rainfall: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Enforces physical constraints on rainfall predictions:
    1. Rainfall cannot be negative (clipped to 0.0).
    2. Upper physical threshold for daily rainfall in India (clipped to 1500 mm).
    3. Truncation of tiny drizzle noise (< 0.1 mm to 0.0).
    """
    arr = np.asarray(rainfall, dtype=float)
    arr = np.maximum(arr, 0.0)
    arr = np.minimum(arr, 1500.0)
    # Zero out sub-physical numerical noise
    arr[arr < 0.1] = 0.0
    return np.round(arr, 2)

def transform_log1p(rainfall: np.ndarray) -> np.ndarray:
    """Applies log1p transformation to stabilize variance in heavily skewed rainfall."""
    return np.log1p(np.maximum(rainfall, 0.0))

def invert_expm1(transformed: np.ndarray) -> np.ndarray:
    """Inverts log1p transformation and ensures non-negativity."""
    inverted = np.expm1(transformed)
    return apply_physical_constraints(inverted)

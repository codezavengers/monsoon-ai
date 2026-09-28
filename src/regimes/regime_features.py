"""
Regime feature representation and encoding utilities.
"""

from typing import List, Dict
import numpy as np
from src.regimes.rules import REGIME_NAMES

def get_regime_to_index_map() -> Dict[str, int]:
    """Returns mapping from regime name to integer index."""
    return {name: i for i, name in enumerate(REGIME_NAMES)}

def encode_regime_one_hot(regime: str) -> np.ndarray:
    """One-hot encodes a single regime name into a vector of length len(REGIME_NAMES)."""
    mapping = get_regime_to_index_map()
    vec = np.zeros(len(REGIME_NAMES), dtype=np.float32)
    idx = mapping.get(regime, 0)
    vec[idx] = 1.0
    return vec

def encode_regimes_batch(regimes: List[str]) -> np.ndarray:
    """Encodes a list of regime strings into a 2D one-hot matrix."""
    mapping = get_regime_to_index_map()
    matrix = np.zeros((len(regimes), len(REGIME_NAMES)), dtype=np.float32)
    for i, reg in enumerate(regimes):
        idx = mapping.get(reg, 0)
        matrix[i, idx] = 1.0
    return matrix

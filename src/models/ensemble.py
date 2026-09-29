"""
Hybrid Regime Ensemble Model.
Blends global ML post-processing with regime-specific corrections.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class HybridRegimeEnsemble:
    """Ensemble combining global baseline ML and regime-specialized models."""
    def __init__(self, global_weight: float = 0.35, regime_weight: float = 0.65):
        self.global_weight = global_weight
        self.regime_weight = regime_weight

    def predict(self, global_preds: np.ndarray, regime_preds: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.global_weight * global_preds + self.regime_weight * regime_preds)

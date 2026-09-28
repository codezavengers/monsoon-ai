"""
Baseline models:
1. Raw NWP (unmodified numerical weather prediction).
2. Global Mean Bias Correction.
3. Standard Global ML Post-Processor (without regime stratification).
"""

from typing import Union
import numpy as np
from src.models.correction import apply_physical_constraints, transform_log1p, invert_expm1

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class RawNWPBaseline:
    """Baseline 1: Uses raw NWP forecast without post-processing."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RawNWPBaseline":
        return self
        
    def predict(self, X: np.ndarray, raw_nwp_idx: int = 0) -> np.ndarray:
        """Returns raw NWP rainfall directly."""
        return apply_physical_constraints(X[:, raw_nwp_idx])

class GlobalMeanBiasCorrection:
    """Baseline 2: Global additive mean bias correction."""
    def __init__(self):
        self.mean_bias = 0.0
        
    def fit(self, raw_nwp: np.ndarray, y_obs: np.ndarray) -> "GlobalMeanBiasCorrection":
        diff = y_obs - raw_nwp
        self.mean_bias = float(np.mean(diff))
        return self
        
    def predict(self, raw_nwp: np.ndarray) -> np.ndarray:
        corrected = raw_nwp + self.mean_bias
        return apply_physical_constraints(corrected)

class GlobalMLPostProcessor:
    """
    Standard ML Post-Processor trained on all features globally,
    without any weather-regime classification or stratification.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GlobalMLPostProcessor":
        y_log = transform_log1p(y)
        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X, y_log)
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            # Fallback to NWP rain (column 0)
            return apply_physical_constraints(X[:, 0])
        pred_log = self.model.predict(X)
        return invert_expm1(pred_log)

"""
Global (regime-agnostic) bias correction model.
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class GlobalBiasCorrection:
    """Standard global mean and linear bias correction."""
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.mean_bias: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        raw = X["rainfall_nwp"].values
        obs = np.asarray(y)
        self.mean_bias = float(np.mean(raw - obs))
        self.model.fit(X[["rainfall_nwp", "humidity", "elevation", "pressure"]], obs)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_fitted:
            preds = self.model.predict(X[["rainfall_nwp", "humidity", "elevation", "pressure"]])
            return np.maximum(0.0, preds)
        return np.maximum(0.0, X["rainfall_nwp"].values - self.mean_bias)

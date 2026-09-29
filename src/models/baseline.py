"""
Baseline post-processing models: Raw NWP and Climatological Mean.
"""

from typing import Union
import numpy as np
import pandas as pd

class RawNWPBaseline:
    """Identity model representing direct raw numerical weather prediction output."""
    def fit(self, X, y=None):
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return np.maximum(0.0, X["rainfall_nwp"].values)
        return np.maximum(0.0, np.asarray(X)[:, 0])

class ClimatologyBaseline:
    """Empirical climatological baseline using historical training mean."""
    def __init__(self):
        self.climatology_mean: float = 14.8

    def fit(self, X, y: Union[pd.Series, np.ndarray]):
        self.climatology_mean = float(np.mean(y))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.climatology_mean)

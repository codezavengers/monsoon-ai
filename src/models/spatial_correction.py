"""
Spatial Correction and Predictive Displacement for Monsoon Rainfall Grids.
Compensates for numerical advection lag, orographic windward/leeward displacement,
and grid-cell centroid errors.
"""

import os
import pickle
from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.ndimage as ndimage

class PredictiveDisplacementModel:
    """
    Predicts spatial shift vectors (delta_lat, delta_lon) of precipitation clusters
    based on atmospheric steering flow (u850, v850), vertical velocity, and topography.
    """
    def __init__(self):
        self.is_fitted: bool = False
        self.mean_displacement: float = 0.0
        self.model_weights_lat: np.ndarray = np.zeros(4)
        self.model_weights_lon: np.ndarray = np.zeros(4)
        self.feature_means: np.ndarray = np.zeros(4)
        self.feature_stds: np.ndarray = np.ones(4)

    def fit(self, features: np.ndarray, delta_lats: np.ndarray, delta_lons: np.ndarray):
        """Fits ridge regression on kinematic steering features."""
        X = np.asarray(features)
        y_lat = np.asarray(delta_lats)
        y_lon = np.asarray(delta_lons)

        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.maximum(1e-5, np.std(X, axis=0))
        X_norm = (X - self.feature_means) / self.feature_stds

        # Regularized pseudo-inverse
        reg = 10.0 * np.eye(X.shape[1])
        xtx = X_norm.T @ X_norm + reg
        self.model_weights_lat = np.linalg.solve(xtx, X_norm.T @ y_lat)
        self.model_weights_lon = np.linalg.solve(xtx, X_norm.T @ y_lon)

        self.mean_displacement = float(np.mean(np.sqrt(y_lat**2 + y_lon**2)))
        self.is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        if not self.is_fitted:
            return 0.0, 0.0
        X = np.asarray(features).flatten()
        if len(X) < len(self.feature_means):
            X = np.pad(X, (0, len(self.feature_means) - len(X)))
        X_norm = (X[:4] - self.feature_means) / self.feature_stds
        d_lat = float(X_norm @ self.model_weights_lat)
        d_lon = float(X_norm @ self.model_weights_lon)
        return float(np.clip(d_lat, -1.0, 1.0)), float(np.clip(d_lon, -1.0, 1.0))

    def save(self, filepath: str = "models/predictive_displacement.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/predictive_displacement.pkl") -> "PredictiveDisplacementModel":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

class SpatialRainfallPostProcessor:
    """
    2D gridded post-processor:
    1. Spatial advective shift by predicted displacement (dy, dx).
    2. Adaptive gaussian smoothing to reduce localized double-penalty error.
    """
    def __init__(self, displacement_model: Optional[PredictiveDisplacementModel] = None):
        self.displacement_model = displacement_model or PredictiveDisplacementModel()

    def process_grid(
        self,
        raw_grid: np.ndarray,
        steering_features: Optional[np.ndarray] = None,
        sigma_smooth: float = 0.6
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        grid = np.nan_to_num(raw_grid, nan=0.0).copy()
        
        # Predict displacement
        if steering_features is not None and self.displacement_model.is_fitted:
            d_lat, d_lon = self.displacement_model.predict(steering_features)
        else:
            d_lat, d_lon = 0.0, 0.0

        # Pixel shift approximation on 0.25-deg grid
        shift_y = d_lat / 0.25
        shift_x = d_lon / 0.25

        if abs(shift_y) > 0.05 or abs(shift_x) > 0.05:
            shifted = ndimage.shift(grid, shift=(shift_y, shift_x), mode="nearest", order=1)
        else:
            shifted = grid

        # Mild spatial relaxation filter
        smoothed = ndimage.gaussian_filter(shifted, sigma=sigma_smooth)
        # Conserve domain maximum amplitude
        max_orig = np.max(grid)
        max_smooth = np.max(smoothed)
        if max_smooth > 0:
            smoothed = smoothed * (max_orig / max_smooth)

        return np.maximum(0.0, smoothed), (d_lat, d_lon)

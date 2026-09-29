"""
Probabilistic Rainfall Predictor and Quantile Estimator.
Produces calibrated exceedance probabilities (Heavy, Very Heavy, Extreme)
and uncertainty prediction intervals (P10, P50, P90).
"""

import os
import pickle
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from src.data.feature_engineering import FEATURE_NAMES

class ProbabilisticRainfallPredictor:
    """
    Calibrated probabilistic forecasting for high-impact monsoon precipitation thresholds
    and asymmetric pinball loss quantile regression.
    """
    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        random_state: int = 42
    ):
        self.thresholds = thresholds or {
            "heavy_64_5mm": 64.5,
            "very_heavy_115_6mm": 115.6,
            "extreme_204_5mm": 204.5
        }
        self.random_state = random_state
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.quantile_models: Dict[str, GradientBoostingRegressor] = {}
        self.calibration_curves: Dict[str, Any] = {}
        self.is_trained: bool = False
        self.feature_names = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        X_df = X[self.feature_names].copy() if all(f in X.columns for f in self.feature_names) else X.copy()
        X_mat = X_df.values
        y_arr = np.asarray(y)

        # 1. Train and calibrate exceedance classifiers
        for name, thresh in self.thresholds.items():
            y_binary = (y_arr >= thresh).astype(int)
            base_clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
            calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
            calibrated_clf.fit(X_mat, y_binary)
            self.models[name] = calibrated_clf

        # 2. Train quantile models: P10, P50, P90
        quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
        for q_name, alpha in quantiles.items():
            q_model = GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=60,
                max_depth=6,
                random_state=self.random_state
            )
            q_model.fit(X_mat, y_arr)
            self.quantile_models[q_name] = q_model

        self.is_trained = True
        return self

    def predict_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        X_df = X[self.feature_names].copy() if all(f in X.columns for f in self.feature_names) else X.copy()
        X_mat = X_df.values
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict_proba(X_mat)[:, 1]
        return results

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        X_df = X[self.feature_names].copy() if all(f in X.columns for f in self.feature_names) else X.copy()
        X_mat = X_df.values
        return {
            q_name: np.maximum(0.0, model.predict(X_mat))
            for q_name, model in self.quantile_models.items()
        }

    def predict_single(self, feature_dict: Dict[str, Any], point_prediction: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Produces exceedance probabilities and prediction intervals for a single station."""
        if not self.is_trained:
            # Physics-based logistic exceedance
            scale = 14.0
            p_h = float(1.0 / (1.0 + np.exp(-(point_prediction - 64.5) / scale)))
            p_vh = float(1.0 / (1.0 + np.exp(-(point_prediction - 115.6) / scale)))
            p_ext = float(1.0 / (1.0 + np.exp(-(point_prediction - 204.5) / scale)))
            
            p10 = round(max(0.0, point_prediction * 0.75), 1)
            p50 = round(point_prediction, 1)
            p90 = round(point_prediction * 1.35 + 4.0, 1)
            return (
                {"heavy_64_5mm": round(p_h, 3), "very_heavy_115_6mm": round(p_vh, 3), "extreme_204_5mm": round(p_ext, 3)},
                {"p10": p10, "p50": p50, "p90": p90, "spread": round(p90 - p10, 1)}
            )

        vals = np.array([[float(feature_dict.get(f, 0.0)) for f in self.feature_names]])
        probs = {}
        for name, model in self.models.items():
            probs[name] = round(float(model.predict_proba(vals)[0, 1]), 3)

        p10 = round(float(np.maximum(0.0, self.quantile_models["p10"].predict(vals)[0])), 1)
        p50 = round(float(np.maximum(0.0, self.quantile_models["p50"].predict(vals)[0])), 1)
        p90 = round(float(np.maximum(p50, self.quantile_models["p90"].predict(vals)[0])), 1)
        spread = round(p90 - p10, 1)

        return probs, {"p10": p10, "p50": p50, "p90": p90, "spread": spread}

    def save(self, filepath: str = "models/prob_predictor.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/prob_predictor.pkl") -> "ProbabilisticRainfallPredictor":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

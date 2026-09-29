"""
Regime-Aware Machine Learning Post-Processor.
Routes precipitation predictions through regime-specialized regressors with soft mixture weighting.
"""

import os
import pickle
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.regimes.rules import REGIME_NAMES
from src.data.feature_engineering import FEATURE_NAMES

class RegimeSpecificMLPostProcessor:
    """
    Proposed Architecture: Regime-specialized ML models combined via Bayesian/Soft posterior mixture.
    Eliminates boundary discontinuities across spatial regime transitions.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        routing: str = "soft_mixture",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.routing = routing
        self.random_state = random_state
        self.regimes = REGIME_NAMES
        self.feature_names = FEATURE_NAMES

        # Regime-specific models
        self.models: Dict[str, Any] = {}
        self.global_fallback = Ridge(alpha=1.0)
        self.is_trained: bool = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], regimes: Union[pd.Series, List[str]]):
        X_df = X[self.feature_names].copy() if all(f in X.columns for f in self.feature_names) else X.copy()
        y_arr = np.asarray(y)
        reg_arr = np.asarray(regimes)

        self.global_fallback.fit(X_df.values, y_arr)

        for regime in self.regimes:
            mask = (reg_arr == regime)
            if np.sum(mask) >= 15:
                reg_model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                reg_model.fit(X_df.values[mask], y_arr[mask])
                self.models[regime] = reg_model
            else:
                self.models[regime] = self.global_fallback

        self.is_trained = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        regime_probs: Optional[Dict[str, np.ndarray]] = None,
        hard_regimes: Optional[List[str]] = None
    ) -> np.ndarray:
        if not self.is_trained:
            raw = X["rainfall_nwp"].values if "rainfall_nwp" in X.columns else np.zeros(len(X))
            return np.maximum(0.0, raw)

        X_df = X[self.feature_names].copy() if all(f in X.columns for f in self.feature_names) else X.copy()
        X_mat = X_df.values
        n_samples = len(X_df)

        if self.routing == "soft_mixture" and regime_probs is not None:
            combined = np.zeros(n_samples)
            for reg, model in self.models.items():
                p = regime_probs.get(reg, np.zeros(n_samples))
                preds = model.predict(X_mat)
                combined += p * preds
            return np.maximum(0.0, combined)

        # Hard routing fallback
        if hard_regimes is not None:
            results = np.zeros(n_samples)
            for i, r in enumerate(hard_regimes):
                m = self.models.get(r, self.global_fallback)
                results[i] = m.predict(X_mat[i:i+1])[0]
            return np.maximum(0.0, results)

        return np.maximum(0.0, self.global_fallback.predict(X_mat))

    def predict_single(self, feature_dict: Dict[str, Any], regime_details: Dict[str, Any]) -> float:
        """Inference for a single station record using soft regime mixture."""
        if not self.is_trained:
            raw = float(feature_dict.get("rainfall_nwp", feature_dict.get("rainfall", 0.0)))
            return round(max(0.0, raw * 1.08), 1)

        vals = np.array([[float(feature_dict.get(f, 0.0)) for f in self.feature_names]])
        probs = regime_details.get("probabilities", {})

        weighted_pred = 0.0
        total_p = 0.0
        for reg, model in self.models.items():
            p = float(probs.get(reg, 0.0))
            if p > 0.0:
                p_val = model.predict(vals)[0]
                weighted_pred += p * p_val
                total_p += p

        if total_p > 0.0:
            final_val = weighted_pred / total_p
        else:
            final_val = self.global_fallback.predict(vals)[0]

        return round(float(np.maximum(0.0, final_val)), 1)

    def save(self, filepath: str = "models/regime_ml_model.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/regime_ml_model.pkl") -> "RegimeSpecificMLPostProcessor":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

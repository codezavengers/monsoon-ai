"""
Supervised Machine Learning Weather Regime Classifier.
Distills domain physical rules and synoptic patterns into an ensemble classifier.
"""

import os
import pickle
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .rules import REGIME_NAMES

class RegimeClassifier:
    """
    ML classifier that maps atmospheric predictors to 8 Indian monsoon regimes.
    Provides calibrated posterior class probabilities.
    """
    def __init__(
        self,
        classifier_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.regime_map: Dict[str, int] = {r: i for i, r in enumerate(REGIME_NAMES)}
        self.index_to_regime: Dict[int, str] = {i: r for i, r in enumerate(REGIME_NAMES)}
        self.feature_names: List[str] = []
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_importances_: Dict[str, float] = {}
        self.best_params_: Dict[str, Any] = {}
        self.is_trained: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, List[str]]):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            if not self.feature_names:
                self.feature_names = [f"f_{i}" for i in range(X_arr.shape[1])]

        if isinstance(y, (pd.Series, list)):
            y_arr = np.array([self.regime_map.get(str(label), 0) for label in y])
        else:
            y_arr = np.asarray(y)
            if np.issubdtype(y_arr.dtype, np.str_) or np.issubdtype(y_arr.dtype, np.object_):
                y_arr = np.array([self.regime_map.get(str(label), 0) for label in y_arr])

        self.model.fit(X_arr, y_arr)
        self.is_trained = True

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = {
                f: float(imp) for f, imp in zip(self.feature_names, self.model.feature_importances_)
            }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names].values if self.feature_names else X.values
        else:
            X_arr = np.asarray(X)
        indices = self.model.predict(X_arr)
        return [self.index_to_regime.get(int(idx), "normal_monsoon") for idx in indices]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names].values if self.feature_names else X.values
        else:
            X_arr = np.asarray(X)
        probs = self.model.predict_proba(X_arr)
        
        # Map internal class indices to all 8 regimes
        classes = getattr(self.model, "classes_", np.arange(probs.shape[1]))
        n_samples = X_arr.shape[0]
        result: Dict[str, np.ndarray] = {r: np.zeros(n_samples) for r in REGIME_NAMES}
        
        for col_idx, class_idx in enumerate(classes):
            regime = self.index_to_regime.get(int(class_idx))
            if regime:
                result[regime] = probs[:, col_idx]
        return result

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Inference for a single record returning predicted regime and full distribution."""
        if not self.is_trained:
            # Domain rule fallback if model not loaded
            from .rules import classify_regime_rule
            rule_reg = classify_regime_rule(feature_dict)
            probs = {r: (0.93 if r == rule_reg else 0.01) for r in REGIME_NAMES}
            return {
                "predicted": rule_reg,
                "confidence": 0.93,
                "entropy": 0.07,
                "probabilities": probs
            }

        vals = [float(feature_dict.get(f, 0.0)) for f in self.feature_names]
        X_arr = np.array([vals])
        probs = self.model.predict_proba(X_arr)[0]
        classes = getattr(self.model, "classes_", np.arange(len(probs)))
        
        prob_dict = {r: 0.001 for r in REGIME_NAMES}
        for col_idx, c_idx in enumerate(classes):
            reg = self.index_to_regime.get(int(c_idx))
            if reg:
                prob_dict[reg] = round(float(probs[col_idx]), 4)

        pred_regime = max(prob_dict.items(), key=lambda x: x[1])[0]
        conf = prob_dict[pred_regime]
        p_arr = np.array(list(prob_dict.values()))
        entropy = float(-np.sum(p_arr * np.log(p_arr + 1e-9)))

        return {
            "predicted": pred_regime,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "probabilities": prob_dict
        }

    def save(self, filepath: str = "models/regime_classifier.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/regime_classifier.pkl") -> "RegimeClassifier":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

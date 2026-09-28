"""
Hybrid Regime-Aware Ensemble Model.
Concatenates meteorological predictors with one-hot encoded regime features
and regime class probability distributions to allow continuous regime awareness.
"""

from typing import List, Dict, Any, Union
import numpy as np

from src.models.correction import apply_physical_constraints, transform_log1p, invert_expm1
from src.regimes.regime_features import encode_regimes_batch

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class HybridRegimeEnsembleModel:
    """
    Hybrid Model:
    Meteorological Features + One-Hot Regimes + Spatial + Temporal -> ML Post-Processor
    """
    def __init__(self, n_estimators: int = 120, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
    def _prepare_features(self, X: np.ndarray, regimes: List[str]) -> np.ndarray:
        regime_onehot = encode_regimes_batch(regimes)
        return np.hstack([X, regime_onehot])
        
    def fit(self, X: np.ndarray, y: np.ndarray, regimes: List[str]) -> "HybridRegimeEnsembleModel":
        X_augmented = self._prepare_features(X, regimes)
        y_log = transform_log1p(y)
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                subsample=0.85
            )
            self.model.fit(X_augmented, y_log)
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray, regimes: List[str]) -> np.ndarray:
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            return apply_physical_constraints(X[:, 0])
            
        X_augmented = self._prepare_features(X, regimes)
        preds_log = self.model.predict(X_augmented)
        return invert_expm1(preds_log)

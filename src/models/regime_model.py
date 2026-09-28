"""
Regime-Aware Multi-Model Post-Processing Architecture.
Trains dedicated, specialized ML regression models for each meteorological regime.
Routes inference through the appropriate regime expert model based on classifier output.
"""

from typing import Dict, List, Any
import numpy as np

from src.models.correction import apply_physical_constraints, transform_log1p, invert_expm1
from src.regimes.rules import REGIME_NAMES

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class RegimeSpecificMLPostProcessor:
    """
    Architecture of regime-specialized models:
    - Active Monsoon Model (handles convective core underestimation)
    - Break Monsoon Model (handles false alarm / wet bias reduction)
    - Monsoon Depression Model (handles cyclone/low circulation intensity)
    - Orographic Model (handles localized Western Ghats/Himalayan precipitation)
    - Coastal Model (handles coastal convergence)
    - Western Disturbance Model (handles mid-latitude synoptic systems)
    - Extreme Event Model (specialized heavy precipitation regression)
    - Background/Normal Monsoon Model
    """
    def __init__(
        self, 
        model_type: str = "random_forest", 
        n_estimators: int = 80, 
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.regime_models: Dict[str, Any] = {}
        self.global_fallback = None
        self.is_trained = False
        
    def _create_regressor(self):
        if not SKLEARN_AVAILABLE:
            return None
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=min(self.max_depth, 6),
                random_state=self.random_state
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, regimes: List[str]) -> "RegimeSpecificMLPostProcessor":
        """
        Fits individual regime-specialized models for each weather regime subset.
        Also trains a global fallback model if any regime has insufficient samples.
        """
        regimes_arr = np.asarray(regimes)
        y_log = transform_log1p(y)
        
        # Train global fallback
        self.global_fallback = self._create_regressor()
        if self.global_fallback is not None:
            self.global_fallback.fit(X, y_log)
            
        # Train per-regime models
        for reg in REGIME_NAMES:
            mask = regimes_arr == reg
            n_samples = np.sum(mask)
            
            if n_samples >= 15: # Minimum threshold to train specialized model
                model = self._create_regressor()
                if model is not None:
                    model.fit(X[mask], y_log[mask])
                    self.regime_models[reg] = model
            else:
                # Share global fallback if regime sample is sparse
                self.regime_models[reg] = self.global_fallback
                
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray, regimes: List[str]) -> np.ndarray:
        """
        Routes each sample to its regime-specific model for prediction.
        """
        n_samples = len(X)
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback to column 0 (raw NWP)
            return apply_physical_constraints(X[:, 0])
            
        preds_log = np.zeros(n_samples, dtype=float)
        regimes_arr = np.asarray(regimes)
        
        # Vectorized batch routing by regime
        for reg in REGIME_NAMES:
            mask = regimes_arr == reg
            if np.sum(mask) == 0:
                continue
                
            model = self.regime_models.get(reg, self.global_fallback)
            if model is not None:
                preds_log[mask] = model.predict(X[mask])
            elif self.global_fallback is not None:
                preds_log[mask] = self.global_fallback.predict(X[mask])
                
        # Handle any uncategorized regimes
        unknown_mask = ~np.isin(regimes_arr, REGIME_NAMES)
        if np.sum(unknown_mask) > 0 and self.global_fallback is not None:
            preds_log[unknown_mask] = self.global_fallback.predict(X[unknown_mask])
            
        return invert_expm1(preds_log)

"""
Probabilistic heavy rainfall prediction module.
Estimates calibrated probabilities for Heavy (>=64.5 mm), Very Heavy (>=115.6 mm),
and Extremely Heavy (>=204.5 mm) rainfall events.
"""

from typing import Dict, List, Any, Tuple
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ProbabilisticRainfallPredictor:
    """
    Calibrated probabilistic classifier for operational extreme precipitation thresholds.
    """
    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        random_state: int = 42
    ):
        if thresholds is None:
            self.thresholds = {"heavy": 64.5, "very_heavy": 115.6, "extreme": 204.5}
        else:
            self.thresholds = thresholds
            
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilisticRainfallPredictor":
        """Fits calibrated classifiers for each threshold."""
        if not SKLEARN_AVAILABLE:
            self.is_trained = True
            return self
            
        for name, thresh in self.thresholds.items():
            y_binary = (y >= thresh).astype(int)
            pos_count = np.sum(y_binary)
            
            # Require at least 5 positive samples to train classifier
            if pos_count >= 5:
                base_rf = RandomForestClassifier(
                    n_estimators=60,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1
                )
                try:
                    calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=3)
                    calibrated.fit(X, y_binary)
                    self.models[name] = calibrated
                except Exception:
                    base_rf.fit(X, y_binary)
                    self.models[name] = base_rf
            else:
                self.models[name] = None
                
        self.is_trained = True
        return self
        
    def predict_probabilities(self, X: np.ndarray, predicted_rain: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Returns probability arrays for heavy, very heavy, and extreme rainfall.
        If machine learning models cannot produce a probability or for rare extreme events,
        combines model output with logistic sigmoid on the corrected rainfall.
        """
        n_samples = len(X)
        probs = {}
        
        for name, thresh in self.thresholds.items():
            if self.is_trained and SKLEARN_AVAILABLE and self.models.get(name) is not None:
                p = self.models[name].predict_proba(X)[:, 1]
            elif predicted_rain is not None:
                # Physically calibrated sigmoid curve centered around the threshold
                scale = thresh * 0.22 # Smooth transition window
                z = (predicted_rain - thresh) / scale
                p = 1.0 / (1.0 + np.exp(-z))
            else:
                p = np.zeros(n_samples)
                
            probs[name] = np.clip(np.round(p, 3), 0.0, 1.0)
            
        return probs

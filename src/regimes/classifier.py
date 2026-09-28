"""
Machine Learning Weather Regime Classifier.
Trains supervised classifier on meteorological & geospatial variables,
evaluates precision/recall/F1, and outputs feature importances.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from src.regimes.rules import REGIME_NAMES, classify_regime_rule
from src.regimes.regime_features import get_regime_to_index_map
from src.data.feature_engineering import FEATURE_NAMES

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class RegimeClassifier:
    """
    Supervised Regime Classifier.
    Trains on meteorological predictors to identify prevailing weather regimes.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.regime_map = get_regime_to_index_map()
        self.index_to_regime = {v: k for k, v in self.regime_map.items()}
        self.feature_names = FEATURE_NAMES
        self.model = None
        self.feature_importances_ = None
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: List[str]) -> "RegimeClassifier":
        """Fits the classifier on feature matrix X and string regime labels y."""
        y_encoded = np.array([self.regime_map.get(label, 0) for label in y])
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced"
            )
            self.model.fit(X, y_encoded)
            self.feature_importances_ = self.model.feature_importances_
        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> List[str]:
        """Predicts regime labels for feature matrix X."""
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            # Fallback
            return [REGIME_NAMES[0] for _ in range(len(X))]
            
        preds_encoded = self.model.predict(X)
        return [self.index_to_regime.get(idx, "normal_monsoon") for idx in preds_encoded]
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities."""
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            n_classes = len(REGIME_NAMES)
            uniform = np.full((len(X), n_classes), 1.0 / n_classes)
            return uniform
            
        probas = self.model.predict_proba(X)
        # Ensure shape matches total number of regimes
        full_probas = np.zeros((len(X), len(REGIME_NAMES)))
        classes = self.model.classes_
        for i, c in enumerate(classes):
            full_probas[:, c] = probas[:, i]
        return full_probas
        
    def evaluate(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        """Evaluates classifier accuracy, precision, recall, F1, and confusion matrix."""
        y_encoded = np.array([self.regime_map.get(label, 0) for label in y])
        preds = self.predict(X)
        preds_encoded = np.array([self.regime_map.get(label, 0) for label in preds])
        
        if SKLEARN_AVAILABLE:
            acc = float(accuracy_score(y_encoded, preds_encoded))
            labels_all = list(range(len(REGIME_NAMES)))
            cm = confusion_matrix(y_encoded, preds_encoded, labels=labels_all)
            report = classification_report(y_encoded, preds_encoded, labels=labels_all, target_names=REGIME_NAMES, output_dict=True, zero_division=0)
        else:
            acc = float(np.mean(y_encoded == preds_encoded))
            cm = np.zeros((len(REGIME_NAMES), len(REGIME_NAMES)))
            report = {}
            
        feat_imp = []
        if self.feature_importances_ is not None:
            for name, imp in zip(self.feature_names, self.feature_importances_):
                feat_imp.append({"feature": name, "importance": round(float(imp), 4)})
            feat_imp.sort(key=lambda x: x["importance"], reverse=True)
            
        return {
            "accuracy": round(acc, 3),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importance": feat_imp
        }

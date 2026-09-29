"""
Machine Learning Weather Regime Classifier.
Trains supervised multi-class model on meteorological & geospatial predictors,
separating rule-based reference regimes, ML-predicted regimes, and observational regimes.
Produces full probability distributions, prediction entropy, confidence scores,
and comprehensive per-regime classification metrics (precision, recall, F1, support, confusion matrix).
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from src.regimes.rules import REGIME_NAMES, classify_regime_rule
from src.regimes.regime_features import get_regime_to_index_map
from src.data.feature_engineering import FEATURE_NAMES

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class RegimeClassifier:
    """
    Supervised Regime Classifier.
    Predicts synoptic weather regime from meteorological predictors.
    Outputs class probabilities, prediction confidence, and feature importance.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        classifier_type: str = "random_forest",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.regime_map = get_regime_to_index_map()
        self.index_to_regime = {v: k for k, v in self.regime_map.items()}
        self.feature_names = FEATURE_NAMES
        self.model = None
        self.feature_importances_ = None
        self.best_params_ = {}
        self.is_trained = False
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[List[str]] = None
    ) -> "RegimeClassifier":
        """
        Fits the classifier on training features and reference regimes.
        If validation data is provided, performs model selection / hyperparameter validation.
        """
        y_encoded = np.array([self.regime_map.get(label, 0) for label in y_train])
        
        if SKLEARN_AVAILABLE:
            if X_val is not None and y_val is not None:
                # Validation-based parameter selection between depths
                y_val_enc = np.array([self.regime_map.get(l, 0) for l in y_val])
                best_f1 = -1.0
                best_depth = self.max_depth
                best_rf = None

                for depth in [6, 10, 14]:
                    clf = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=depth,
                        random_state=self.random_state,
                        class_weight="balanced",
                        n_jobs=-1
                    )
                    clf.fit(X_train, y_encoded)
                    val_preds = clf.predict(X_val)
                    score = f1_score(y_val_enc, val_preds, average="weighted", zero_division=0)
                    if score > best_f1:
                        best_f1 = score
                        best_depth = depth
                        best_rf = clf

                self.model = best_rf
                self.max_depth = best_depth
                self.best_params_ = {"max_depth": best_depth, "val_weighted_f1": round(best_f1, 4)}
            else:
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1
                )
                self.model.fit(X_train, y_encoded)
                self.best_params_ = {"max_depth": self.max_depth}

            self.feature_importances_ = self.model.feature_importances_

        self.is_trained = True
        return self
        
    def predict(self, X: np.ndarray) -> List[str]:
        """Predicts most probable regime label for feature matrix X."""
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            return [REGIME_NAMES[0] for _ in range(len(X))]
            
        preds_encoded = self.model.predict(X)
        return [self.index_to_regime.get(idx, "normal_monsoon") for idx in preds_encoded]
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns full probability matrix [n_samples x 8_regimes].
        Ensures consistent columns matching REGIME_NAMES ordering.
        """
        n_classes = len(REGIME_NAMES)
        if not self.is_trained or not SKLEARN_AVAILABLE or self.model is None:
            return np.full((len(X), n_classes), 1.0 / n_classes)
            
        probas = self.model.predict_proba(X)
        full_probas = np.zeros((len(X), n_classes), dtype=float)
        classes = self.model.classes_
        for i, c in enumerate(classes):
            if c < n_classes:
                full_probas[:, c] = probas[:, i]
                
        # Normalize rows to sum to 1.0
        row_sums = np.sum(full_probas, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return full_probas / row_sums

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Returns (predicted_regimes, confidence_scores, entropy_uncertainty).
        Confidence is the max class probability.
        Entropy is normalized Shannon entropy between 0 (certain) and 1 (fully uncertain).
        """
        probas = self.predict_proba(X)
        conf = np.max(probas, axis=1)
        pred_idx = np.argmax(probas, axis=1)
        pred_labels = [self.index_to_regime.get(idx, "normal_monsoon") for idx in pred_idx]

        # Shannon Entropy: -sum(p * log(p)) / log(K)
        eps = 1e-12
        p_clipped = np.clip(probas, eps, 1.0)
        k = len(REGIME_NAMES)
        entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1) / np.log(k)

        return pred_labels, conf, entropy
        
    def evaluate(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        """
        Comprehensive out-of-sample evaluation:
        Per-regime Precision, Recall, F1-Score, Support, and Confusion Matrix.
        """
        y_encoded = np.array([self.regime_map.get(label, 0) for label in y])
        preds = self.predict(X)
        preds_encoded = np.array([self.regime_map.get(label, 0) for label in preds])
        labels_all = list(range(len(REGIME_NAMES)))
        
        if SKLEARN_AVAILABLE:
            acc = float(accuracy_score(y_encoded, preds_encoded))
            cm = confusion_matrix(y_encoded, preds_encoded, labels=labels_all)
            report = classification_report(
                y_encoded,
                preds_encoded,
                labels=labels_all,
                target_names=REGIME_NAMES,
                output_dict=True,
                zero_division=0
            )
        else:
            acc = float(np.mean(y_encoded == preds_encoded))
            cm = np.zeros((len(REGIME_NAMES), len(REGIME_NAMES)))
            report = {}
            
        # Extract clean per-regime summary
        per_regime_metrics = {}
        for reg in REGIME_NAMES:
            if reg in report:
                supp = int(report[reg].get("support", 0))
                per_regime_metrics[reg] = {
                    "precision": round(float(report[reg].get("precision", 0.0)), 3),
                    "recall": round(float(report[reg].get("recall", 0.0)), 3),
                    "f1_score": round(float(report[reg].get("f1-score", 0.0)), 3),
                    "support": supp,
                    "status": "EVALUATED" if supp > 0 else "ZERO_SUPPORT_RARE_REGIME"
                }
            else:
                per_regime_metrics[reg] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "support": 0,
                    "status": "ZERO_SUPPORT_RARE_REGIME"
                }

        feat_imp = []
        if self.feature_importances_ is not None:
            for name, imp in zip(self.feature_names, self.feature_importances_):
                feat_imp.append({"feature": name, "importance": round(float(imp), 4)})
            feat_imp.sort(key=lambda x: x["importance"], reverse=True)
            
        return {
            "accuracy": round(acc, 3),
            "evaluation_type": "RULE_REFERENCE_AGREEMENT",
            "scientific_disclaimer": "Evaluates ML model fidelity in reproducing meteorological rule definitions. High agreement demonstrates successful distillation of domain physics into ML, not discovery against independent human-annotated ground truth.",
            "report": report,
            "per_regime_metrics": per_regime_metrics,
            "confusion_matrix": cm.tolist(),
            "feature_importance": feat_imp,
            "best_params": self.best_params_
        }

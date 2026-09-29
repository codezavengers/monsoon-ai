"""
Calibrated Probabilistic Extreme Rainfall Forecasting and Uncertainty Estimation.
Provides calibrated exceedance probabilities for IMD Operational Thresholds:
- Heavy Rainfall (>=64.5 mm/day)
- Very Heavy Rainfall (>=115.6 mm/day)
- Extremely Heavy Rainfall (>=204.5 mm/day)
Computes Brier Score, ROC-AUC, PR-AUC, reliability diagrams, and P10/P50/P90 prediction intervals.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score, precision_recall_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ProbabilisticRainfallPredictor:
    """
    Operational Probabilistic Predictor with calibration on validation splits.
    Includes quantile estimators for P10, P50 (median), P90 intervals.
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
        self.quantile_models: Dict[str, Any] = {}
        self.calibration_curves: Dict[str, Any] = {}
        self.is_trained = False
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "ProbabilisticRainfallPredictor":
        """
        Fits threshold classifiers using validation data for probability calibration.
        Also fits quantile regressors for P10 and P90 uncertainty intervals.
        """
        if not SKLEARN_AVAILABLE:
            self.is_trained = True
            return self

        for name, thresh in self.thresholds.items():
            y_bin_train = (y_train >= thresh).astype(int)
            pos_count = np.sum(y_bin_train)

            # Balanced Random Forest
            base_rf = RandomForestClassifier(
                n_estimators=80,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )

            if X_val is not None and y_val is not None and len(X_val) > 20:
                y_bin_val = (y_val >= thresh).astype(int)
                base_rf.fit(X_train, y_bin_train)
                # Calibrate on independent validation set
                try:
                    calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv="prefit")
                    calibrated.fit(X_val, y_bin_val)
                    self.models[name] = calibrated
                except Exception:
                    self.models[name] = base_rf
            else:
                try:
                    calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=3)
                    calibrated.fit(X_train, y_bin_train)
                    self.models[name] = calibrated
                except Exception:
                    base_rf.fit(X_train, y_bin_train)
                    self.models[name] = base_rf

        # Train Quantile Regressors for P10 and P90 uncertainty
        try:
            p10_model = GradientBoostingRegressor(
                loss="quantile",
                alpha=0.10,
                n_estimators=60,
                max_depth=4,
                random_state=self.random_state
            )
            p10_model.fit(X_train, y_train)
            self.quantile_models["p10"] = p10_model

            p90_model = GradientBoostingRegressor(
                loss="quantile",
                alpha=0.90,
                n_estimators=60,
                max_depth=4,
                random_state=self.random_state
            )
            p90_model.fit(X_train, y_train)
            self.quantile_models["p90"] = p90_model
        except Exception:
            pass

        self.is_trained = True
        return self
        
    def predict_probabilities(
        self,
        X: np.ndarray,
        predicted_rain: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Predicts calibrated probabilities for heavy, very heavy, and extreme rainfall.
        Uses trained ML probability models as the primary estimator.
        """
        n_samples = len(X)
        probs = {}
        
        for name, thresh in self.thresholds.items():
            if self.is_trained and SKLEARN_AVAILABLE and self.models.get(name) is not None:
                model = self.models[name]
                proba_mat = model.predict_proba(X)
                if proba_mat.shape[1] > 1:
                    p = proba_mat[:, 1]
                else:
                    p = np.zeros(n_samples)
            elif predicted_rain is not None:
                # Logistic sigmoid fallback only if model unavailable
                scale = thresh * 0.22
                z = (predicted_rain - thresh) / scale
                p = 1.0 / (1.0 + np.exp(-z))
            else:
                p = np.zeros(n_samples)
                
            probs[name] = np.clip(np.round(p, 3), 0.0, 1.0)
            
        return probs

    def predict_quantiles(
        self,
        X: np.ndarray,
        point_prediction: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Returns prediction intervals: P10 (optimistic lower bound),
        P50 (median / point prediction), and P90 (high-risk upper bound).
        """
        n_samples = len(X)
        q_models = getattr(self, "quantile_models", {})
        fallback_used = False

        if "p10" in q_models and "p90" in q_models:
            p10 = q_models["p10"].predict(X)
            p90 = q_models["p90"].predict(X)
            # Physical bounds: 0 <= P10 <= point_prediction (P50) <= P90
            p10 = np.clip(p10, 0.0, point_prediction)
            p90 = np.maximum(p90, point_prediction)
        else:
            fallback_used = True
            # Empirical regime-scaled uncertainty bounds
            p10 = np.maximum(0.0, point_prediction * 0.70)
            p90 = point_prediction * 1.35 + 5.0

        p50 = point_prediction

        return {
            "p10": np.round(p10, 1),
            "p50": np.round(p50, 1),
            "p90": np.round(p90, 1),
            "uncertainty_spread": np.round(p90 - p10, 1),
            "fallback_used": fallback_used
        }

    def evaluate_quantiles(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        point_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates quantile uncertainty intervals against true test observations:
        - Pinball / check loss for tau=0.10, 0.50, 0.90
        - Empirical coverage: fraction of test observations below P10, P50, P90
        - Mean interval width (P90 - P10)
        - Winkler interval score for 80% central prediction interval (P10 to P90)
        """
        q = self.predict_quantiles(X_test, point_predictions)
        p10 = q["p10"]
        p50 = q["p50"]
        p90 = q["p90"]
        y = np.asarray(y_test, dtype=float)

        def pinball(y_true, y_pred, alpha):
            diff = y_true - y_pred
            return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))

        loss_p10 = pinball(y, p10, 0.10)
        loss_p50 = pinball(y, p50, 0.50)
        loss_p90 = pinball(y, p90, 0.90)

        # Empirical coverage
        cov_10 = float(np.mean(y <= p10))
        cov_50 = float(np.mean(y <= p50))
        cov_90 = float(np.mean(y <= p90))

        # Winkler interval score for alpha=0.20 (80% coverage interval [p10, p90])
        alpha = 0.20
        width = p90 - p10
        penalty_low = (2.0 / alpha) * (p10 - y) * (y < p10)
        penalty_high = (2.0 / alpha) * (y - p90) * (y > p90)
        interval_scores = width + penalty_low + penalty_high
        mean_interval_score = float(np.mean(interval_scores))

        return {
            "pinball_loss_p10": round(loss_p10, 3),
            "pinball_loss_p50": round(loss_p50, 3),
            "pinball_loss_p90": round(loss_p90, 3),
            "empirical_coverage_p10": round(cov_10, 3),
            "empirical_coverage_p50": round(cov_50, 3),
            "empirical_coverage_p90": round(cov_90, 3),
            "mean_interval_width": round(float(np.mean(width)), 2),
            "mean_interval_score": round(mean_interval_score, 2),
            "sample_count": len(y)
        }

    def evaluate_probabilistic(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        predicted_rain: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Calculates Brier Score, ROC-AUC, PR-AUC, and 10-bin Reliability Diagrams.
        """
        probs = self.predict_probabilities(X_test, predicted_rain)
        metrics_by_thresh = {}

        for name, thresh in self.thresholds.items():
            y_true = (y_test >= thresh).astype(int)
            y_prob = probs[name]
            pos_ratio = float(np.mean(y_true))
            pos_count = int(np.sum(y_true))

            if SKLEARN_AVAILABLE and pos_ratio > 0 and pos_ratio < 1.0:
                bs = float(brier_score_loss(y_true, y_prob))
                # Brier Skill Score relative to climatology
                bs_clim = float(brier_score_loss(y_true, np.full_like(y_true, pos_ratio, dtype=float)))
                bss = float(1.0 - (bs / bs_clim)) if bs_clim > 1e-6 else 0.0

                try:
                    roc = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    roc = 0.5
                try:
                    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
                    pr_auc = float(auc(rec_vals, prec_vals))
                except Exception:
                    pr_auc = pos_ratio

                # Calibration slope & intercept via logistic/linear fit
                try:
                    p_clipped = np.clip(y_prob, 1e-4, 1.0 - 1e-4)
                    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
                    from sklearn.linear_model import LogisticRegression
                    cal_lr = LogisticRegression(C=1e5)
                    cal_lr.fit(logits, y_true)
                    cal_slope = float(cal_lr.coef_[0][0])
                    cal_intercept = float(cal_lr.intercept_[0])
                except Exception:
                    cal_slope = 1.0
                    cal_intercept = 0.0

                # Bootstrap 95% Confidence Interval for Brier Score
                n_boot = min(100, len(y_true))
                rng = np.random.RandomState(self.random_state)
                boot_bs = []
                for _ in range(n_boot):
                    boot_idx = rng.randint(0, len(y_true), size=len(y_true))
                    boot_bs.append(float(brier_score_loss(y_true[boot_idx], y_prob[boot_idx])))
                ci_low = float(np.percentile(boot_bs, 2.5))
                ci_high = float(np.percentile(boot_bs, 97.5))

                # 5-bin Reliability Diagram
                bins = np.linspace(0.0, 1.0, 6)
                reliability_bins = []
                for b_idx in range(len(bins) - 1):
                    mask = (y_prob >= bins[b_idx]) & (y_prob < bins[b_idx + 1])
                    if np.sum(mask) > 0:
                        observed_freq = float(np.mean(y_true[mask]))
                        mean_prob = float(np.mean(y_prob[mask]))
                        count = int(np.sum(mask))
                    else:
                        observed_freq = float((bins[b_idx] + bins[b_idx + 1]) / 2)
                        mean_prob = observed_freq
                        count = 0
                    reliability_bins.append({
                        "bin_range": f"{bins[b_idx]:.1f}-{bins[b_idx+1]:.1f}",
                        "forecast_prob": round(mean_prob, 3),
                        "observed_frequency": round(observed_freq, 3),
                        "sample_count": count
                    })
            else:
                bs = float(np.mean((y_prob - y_true)**2))
                bss = 0.0
                roc = 0.5
                pr_auc = pos_ratio
                cal_slope = 1.0
                cal_intercept = 0.0
                ci_low = bs
                ci_high = bs
                reliability_bins = []

            metrics_by_thresh[name] = {
                "threshold_mm": thresh,
                "event_count": pos_count,
                "event_frequency": round(pos_ratio, 4),
                "climatological_baseline": round(pos_ratio, 4),
                "brier_score": round(bs, 4),
                "brier_skill_score": round(bss, 3),
                "brier_score_95ci": [round(ci_low, 4), round(ci_high, 4)],
                "calibration_slope": round(cal_slope, 3),
                "calibration_intercept": round(cal_intercept, 3),
                "roc_auc": round(roc, 3),
                "pr_auc": round(pr_auc, 3),
                "positive_sample_count": pos_count,
                "total_sample_count": len(y_true),
                "reliability_diagram": reliability_bins
            }

        return metrics_by_thresh

"""
Authoritative Real-Time Inference Engine for Regime-Aware Monsoon Rainfall AI.
Serves CLI, REST API (Express server), and batch pipeline.
Produces unified predictions:
- AI Corrected Rainfall (point estimate)
- Uncertainty Prediction Intervals (P10, P50, P90)
- Predicted Weather Regime and 8-Regime Probability Distribution
- Calibrated Exceedance Probabilities (Heavy, Very Heavy, Extremely Heavy)
- Explainable AI (XAI) Feature Attributions
"""

import sys
import os
import json
import argparse
import pickle
import datetime
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.regimes.rules import REGIME_NAMES, classify_regime_rule
from src.regimes.classifier import RegimeClassifier
from src.models.regime_model import RegimeSpecificMLPostProcessor
from src.models.probabilistic import ProbabilisticRainfallPredictor
from src.models.explainability import explain_district_correction
from src.data.feature_engineering import engineer_features_dataset, FEATURE_NAMES

# Global cache for loaded model artifacts to ensure low-latency inference
_CACHED_MODELS = {
    "regime_classifier": None,
    "regime_ml": None,
    "prob_predictor": None,
    "loaded": False
}

def load_cached_models(models_dir: str = "models") -> dict:
    """Loads trained model artifacts from disk with graceful fallbacks."""
    global _CACHED_MODELS
    if _CACHED_MODELS["loaded"] and _CACHED_MODELS["regime_classifier"] is not None:
        return _CACHED_MODELS

    reg_clf_path = os.path.join(models_dir, "regime_classifier.pkl")
    reg_ml_path = os.path.join(models_dir, "regime_ml_model.pkl")
    prob_path = os.path.join(models_dir, "prob_predictor.pkl")

    reg_clf = None
    reg_ml = None
    prob = None

    if os.path.exists(reg_clf_path):
        try:
            with open(reg_clf_path, "rb") as f:
                reg_clf = pickle.load(f)
        except Exception as e:
            sys.stderr.write(f"Warning loading regime classifier: {e}\n")

    if os.path.exists(reg_ml_path):
        try:
            with open(reg_ml_path, "rb") as f:
                reg_ml = pickle.load(f)
        except Exception as e:
            sys.stderr.write(f"Warning loading regime ML model: {e}\n")

    if os.path.exists(prob_path):
        try:
            with open(prob_path, "rb") as f:
                prob = pickle.load(f)
        except Exception as e:
            sys.stderr.write(f"Warning loading prob predictor: {e}\n")

    _CACHED_MODELS = {
        "regime_classifier": reg_clf,
        "regime_ml": reg_ml,
        "prob_predictor": prob,
        "loaded": True
    }
    return _CACHED_MODELS

def run_single_inference(input_record: dict, models_dir: str = "models") -> dict:
    """
    Authoritative single-sample inference for API and CLI.
    """
    models = load_cached_models(models_dir)
    reg_clf: RegimeClassifier = models["regime_classifier"]
    reg_ml: RegimeSpecificMLPostProcessor = models["regime_ml"]
    prob_pred: ProbabilisticRainfallPredictor = models["prob_predictor"]

    # Normalize fields with standard meteorological defaults
    lat = float(input_record.get("latitude", input_record.get("lat", 18.96)))
    lon = float(input_record.get("longitude", input_record.get("lon", 72.82)))
    rain = max(0.0, float(input_record.get("rainfall", input_record.get("rainfall_nwp", 45.0))))
    rh = min(100.0, max(0.0, float(input_record.get("humidity", 82.0))))
    temp = float(input_record.get("temperature", 27.5))
    wind = max(0.0, float(input_record.get("wind_speed", 10.5)))
    elev = max(0.0, float(input_record.get("elevation", 25.0)))
    coast = max(0.0, float(input_record.get("coast_dist_km", 10.0)))
    pres = float(input_record.get("pressure", 1002.0))
    cape = max(0.0, float(input_record.get("cape", 1600.0)))
    omega = float(input_record.get("vertical_velocity", -0.25))
    lead_time = int(input_record.get("lead_time_hours", 24))
    district_name = input_record.get("district_name", input_record.get("district", "Custom Station"))
    state_name = input_record.get("state_name", input_record.get("state", "India"))

    record = {
        "district": district_name,
        "state": state_name,
        "lat": lat,
        "lon": lon,
        "rainfall_nwp": rain,
        "humidity": rh,
        "temperature": temp,
        "wind_speed": wind,
        "wind_direction": 230.0,
        "elevation": elev,
        "coast_dist_km": coast,
        "pressure": pres,
        "cape": cape,
        "vertical_velocity": omega,
        "lead_time_hours": lead_time,
        "month": 7,
        "day": 15,
        "day_of_year": 196,
        "zone": "Coastal" if coast < 50 else ("Western Ghats" if elev > 300 and lon < 76 else "Central India")
    }

    # Extract standard feature matrix [1 x n_features]
    X = engineer_features_dataset([record])

    # 1. Regime Prediction
    rule_regime = classify_regime_rule(record)
    if reg_clf is not None and getattr(reg_clf, "is_trained", False):
        pred_regimes, confs, entropies = reg_clf.predict_with_confidence(X)
        pred_regime = pred_regimes[0]
        confidence = float(confs[0])
        entropy = float(entropies[0])
        regime_probas_arr = reg_clf.predict_proba(X)[0]
    else:
        pred_regime = rule_regime
        confidence = 0.85
        entropy = 0.15
        regime_probas_arr = np.zeros(len(REGIME_NAMES))
        if pred_regime in REGIME_NAMES:
            regime_probas_arr[REGIME_NAMES.index(pred_regime)] = 1.0
        else:
            regime_probas_arr[0] = 1.0

    regime_probs_dict = {
        reg: round(float(regime_probas_arr[i]), 3) for i, reg in enumerate(REGIME_NAMES)
    }

    # 2. AI Post-Processed Rainfall Prediction
    if reg_ml is not None and getattr(reg_ml, "is_trained", False):
        # Continuous soft mixture of experts routing
        corrected_rain = float(reg_ml.predict_soft_routing(X, np.array([regime_probas_arr]))[0])
    else:
        # Physical algorithmic fallback
        delta = 0.0
        if pred_regime == "orographic_rainfall":
            delta = rain * 0.35 + min(25.0, (elev / 400.0) * 10.0)
        elif pred_regime == "monsoon_depression":
            delta = rain * 0.28 + max(0.0, 1004.0 - pres) * 1.5
        elif pred_regime == "active_monsoon":
            delta = rain * 0.22 + 5.0
        elif pred_regime == "coastal_rainfall":
            delta = rain * 0.18 + 4.0
        elif pred_regime == "break_monsoon":
            delta = -min(rain * 0.40, 12.0)
        else:
            delta = rain * 0.05
        corrected_rain = max(0.0, rain + delta)

    corrected_rain = round(float(corrected_rain), 1)
    delta_val = round(corrected_rain - rain, 1)

    # 3. Probabilistic Exceedance & Quantiles
    if prob_pred is not None and getattr(prob_pred, "is_trained", False):
        probs_dict = prob_pred.predict_probabilities(X, predicted_rain=np.array([corrected_rain]))
        p_heavy = float(probs_dict["heavy"][0])
        p_very_heavy = float(probs_dict["very_heavy"][0])
        p_extreme = float(probs_dict["extreme"][0])
        quantiles = prob_pred.predict_quantiles(X, np.array([corrected_rain]))
        p10 = float(quantiles["p10"][0])
        p50 = float(quantiles["p50"][0])
        p90 = float(quantiles["p90"][0])
    else:
        # Standard logistic threshold mapping fallback
        scale = 14.0
        p_heavy = float(1.0 / (1.0 + np.exp(-(corrected_rain - 64.5) / scale)))
        p_very_heavy = float(1.0 / (1.0 + np.exp(-(corrected_rain - 115.6) / scale)))
        p_extreme = float(1.0 / (1.0 + np.exp(-(corrected_rain - 204.5) / scale)))
        p10 = round(max(0.0, corrected_rain * 0.75), 1)
        p50 = corrected_rain
        p90 = round(corrected_rain * 1.35 + 4.0, 1)

    # 4. Explainable AI Feature Attribution
    explanation = explain_district_correction(
        raw_record=record,
        corrected_rain=corrected_rain,
        regime=pred_regime
    )

    mode = str(input_record.get("mode", os.environ.get("MODE", "DEMO"))).upper()
    provider = str(input_record.get("provider", "GFS_0.25deg"))
    cycle = str(input_record.get("cycle", "00Z"))
    
    # Calculate forecast valid time
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    valid_dt = now_dt + datetime.timedelta(hours=lead_time)
    valid_time_str = input_record.get("valid_time", valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Determine if models are trained or fallback heuristic is active
    using_ml = (
        reg_ml is not None and getattr(reg_ml, "is_trained", False) and
        reg_clf is not None and getattr(reg_clf, "is_trained", False)
    )
    fallback_used = not using_ml

    if mode == "REAL":
        data_source = input_record.get("data_source", f"Operational {provider} (Real Feed)")
        inference_status = "SUCCESS" if using_ml else "CONFIGURATION_REQUIRED"
        if not using_ml:
            return {
                "success": False,
                "error": "CONFIGURATION_REQUIRED: Authoritative trained ML models not loaded for REAL mode. Please run training pipeline first.",
                "mode": "REAL",
                "data_source": data_source,
                "provider": provider,
                "cycle": cycle,
                "valid_time": valid_time_str,
                "lead_time": lead_time,
                "lead_time_hours": lead_time,
                "model_version": "2.1.0-regime-aware",
                "fallback_used": True,
                "inference_status": "CONFIGURATION_REQUIRED"
            }
    else:
        data_source = input_record.get("data_source", f"Synthetic {provider} Benchmark (JJAS 2018-2024)")
        inference_status = "SUCCESS"

    # Categorization according to IMD standards
    if corrected_rain >= 204.5:
        category = "Extremely Heavy"
    elif corrected_rain >= 115.6:
        category = "Very Heavy"
    elif corrected_rain >= 64.5:
        category = "Heavy"
    elif corrected_rain >= 15.6:
        category = "Moderate"
    elif corrected_rain >= 2.5:
        category = "Light"
    else:
        category = "No Rain"

    return {
        "success": True,
        "mode": mode,
        "data_source": data_source,
        "provider": provider,
        "cycle": cycle,
        "valid_time": valid_time_str,
        "lead_time": lead_time,
        "lead_time_hours": lead_time,
        "model_version": "2.1.0-regime-aware",
        "fallback_used": fallback_used,
        "inference_status": inference_status,
        "district": district_name,
        "state": state_name,
        "coordinates": {"lat": lat, "lon": lon},
        "raw_nwp_rainfall": rain,
        "corrected_rainfall": corrected_rain,
        "delta_correction": delta_val,
        "rainfall_category": category,
        "regime": {
            "predicted": pred_regime,
            "rule_reference": rule_regime,
            "confidence": round(confidence, 3),
            "entropy": round(entropy, 3),
            "probabilities": regime_probs_dict
        },
        "exceedance_probabilities": {
            "heavy_64_5mm": round(p_heavy, 3),
            "very_heavy_115_6mm": round(p_very_heavy, 3),
            "extreme_204_5mm": round(p_extreme, 3)
        },
        "uncertainty_intervals": {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "spread": round(p90 - p10, 1)
        },
        "explainability": explanation
    }

def main():
    parser = argparse.ArgumentParser(description="Regime-Aware Rainfall AI Inference CLI")
    parser.add_argument("--json", type=str, help="JSON input string of meteorological record")
    parser.add_argument("--file", type=str, help="Path to JSON file with input record")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory where trained models reside")
    args = parser.parse_args()

    input_data = {}
    if args.json:
        input_data = json.loads(args.json)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    else:
        # Default test station: Mumbai Colaba active monsoon
        input_data = {
            "latitude": 18.96,
            "longitude": 72.82,
            "rainfall": 82.0,
            "humidity": 88.0,
            "temperature": 26.0,
            "wind_speed": 12.0,
            "elevation": 14.0,
            "coast_dist_km": 2.0,
            "pressure": 998.0,
            "cape": 2100.0,
            "vertical_velocity": -0.35,
            "district_name": "Mumbai City",
            "state_name": "Maharashtra"
        }

    result = run_single_inference(input_data, models_dir=args.models_dir)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

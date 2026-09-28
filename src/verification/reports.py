"""
Verification reporting module.
Generates comprehensive comparative tables across models and per-regime evaluations.
"""

from typing import Dict, List, Any
import numpy as np
from src.verification.deterministic import calculate_deterministic_metrics
from src.verification.categorical import calculate_categorical_metrics
from src.verification.spatial import compute_1d_approx_fss

def generate_model_comparison_report(
    models_dict: Dict[str, np.ndarray], 
    observations: np.ndarray, 
    threshold: float = 64.5
) -> Dict[str, Any]:
    """
    Computes deterministic, categorical, and spatial metrics for each model.
    Models typically include:
    - 'Raw NWP'
    - 'Global Bias Corrected'
    - 'ML (No Regime)'
    - 'Regime-Aware ML'
    """
    results = []
    obs = np.asarray(observations, dtype=float)
    
    for model_name, predictions in models_dict.items():
        fc = np.asarray(predictions, dtype=float)
        det = calculate_deterministic_metrics(fc, obs)
        cat = calculate_categorical_metrics(fc, obs, threshold=threshold)
        fss = compute_1d_approx_fss(fc, obs, threshold=threshold, window_size=5)
        
        results.append({
            "model": model_name,
            "rmse": det["rmse"],
            "mae": det["mae"],
            "bias": det["bias"],
            "correlation": det["correlation"],
            "csi": cat["csi"],
            "ets": cat["ets"],
            "pod": cat["pod"],
            "far": cat["far"],
            "frequency_bias": cat["frequency_bias"],
            "fss": fss,
            "hits": cat["hits"],
            "misses": cat["misses"],
            "false_alarms": cat["false_alarms"]
        })
        
    return {
        "threshold_evaluated_mm": threshold,
        "sample_size": len(obs),
        "comparison_table": results
    }

def generate_regime_wise_verification(
    models_dict: Dict[str, np.ndarray],
    observations: np.ndarray,
    regimes: List[str],
    threshold: float = 64.5
) -> Dict[str, Any]:
    """
    Calculates verification metrics stratified by weather regime.
    Demonstrates whether regime-aware post-processing addresses regime-dependent forecast errors.
    """
    regimes_arr = np.asarray(regimes)
    obs = np.asarray(observations, dtype=float)
    unique_regimes = sorted(list(set(regimes)))
    
    regime_breakdown = []
    
    for reg in unique_regimes:
        mask = regimes_arr == reg
        if np.sum(mask) == 0:
            continue
            
        reg_obs = obs[mask]
        reg_entry = {
            "regime": reg,
            "count": int(np.sum(mask)),
            "obs_mean": round(float(np.mean(reg_obs)), 1),
            "models": {}
        }
        
        for model_name, fc_all in models_dict.items():
            fc_reg = np.asarray(fc_all)[mask]
            det = calculate_deterministic_metrics(fc_reg, reg_obs)
            cat = calculate_categorical_metrics(fc_reg, reg_obs, threshold=threshold)
            reg_entry["models"][model_name] = {
                "rmse": det["rmse"],
                "mae": det["mae"],
                "bias": det["bias"],
                "csi": cat["csi"],
                "pod": cat["pod"],
                "far": cat["far"]
            }
            
        regime_breakdown.append(reg_entry)
        
    return {
        "threshold": threshold,
        "regime_breakdown": regime_breakdown
    }

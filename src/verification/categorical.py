"""
Categorical and Dichotomous Meteorological Verification Metrics.
Computes 2x2 contingency tables, CSI (Threat Score), ETS (Equitable Threat Score),
POD (Hit Rate), FAR (False Alarm Ratio), and Frequency Bias.
"""

from typing import Dict, Any, Union
import numpy as np

def compute_contingency_table(
    forecast: np.ndarray,
    observation: np.ndarray,
    threshold: float = 64.5
) -> Dict[str, int]:
    f = np.asarray(forecast) >= threshold
    o = np.asarray(observation) >= threshold

    hits = int(np.sum(f & o))
    false_alarms = int(np.sum(f & ~o))
    misses = int(np.sum(~f & o))
    correct_negatives = int(np.sum(~f & ~o))

    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct_negatives,
        "total": len(f)
    }

def compute_categorical_scores(
    forecast: np.ndarray,
    observation: np.ndarray,
    threshold: float = 64.5
) -> Dict[str, Any]:
    c = compute_contingency_table(forecast, observation, threshold)
    H = c["hits"]
    F = c["false_alarms"]
    M = c["misses"]
    CN = c["correct_negatives"]
    N = c["total"]

    # POD = H / (H + M)
    pod = H / (H + M) if (H + M) > 0 else 0.0
    # FAR = F / (H + F)
    far = F / (H + F) if (H + F) > 0 else 0.0
    # CSI = H / (H + M + F)
    csi = H / (H + M + F) if (H + M + F) > 0 else 0.0
    # Frequency Bias = (H + F) / (H + M)
    freq_bias = (H + F) / (H + M) if (H + M) > 0 else 1.0

    # ETS = (H - Hr) / (H + M + F - Hr) where Hr = (H + M)*(H + F) / N
    Hr = ((H + M) * (H + F)) / N if N > 0 else 0.0
    ets_denom = (H + M + F - Hr)
    ets = (H - Hr) / ets_denom if ets_denom > 0 else 0.0

    return {
        "hits": H,
        "false_alarms": F,
        "misses": M,
        "correct_negatives": CN,
        "pod": round(pod, 3),
        "far": round(far, 3),
        "csi": round(csi, 3),
        "ets": round(ets, 3),
        "frequency_bias": round(freq_bias, 3)
    }

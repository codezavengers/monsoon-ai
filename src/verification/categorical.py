"""
Categorical (dichotomous) verification metrics for rainfall threshold events.
Calculates 2x2 contingency table, POD, FAR, CSI (Threat Score), ETS (Equitable Threat Score).
"""

from typing import Dict, Any, Union
import numpy as np

def calculate_contingency_table(
    forecast: Union[np.ndarray, list], 
    observation: Union[np.ndarray, list], 
    threshold: float
) -> Dict[str, int]:
    """
    Computes 2x2 contingency table counts for a given threshold event (e.g. >= 64.5 mm).
    - hits (a): forecast >= threshold and obs >= threshold
    - false_alarms (b): forecast >= threshold and obs < threshold
    - misses (c): forecast < threshold and obs >= threshold
    - correct_negatives (d): forecast < threshold and obs < threshold
    """
    fc = np.asarray(forecast, dtype=float)
    obs = np.asarray(observation, dtype=float)
    
    fc_event = fc >= threshold
    obs_event = obs >= threshold
    
    hits = int(np.sum(fc_event & obs_event))
    false_alarms = int(np.sum(fc_event & ~obs_event))
    misses = int(np.sum(~fc_event & obs_event))
    correct_negatives = int(np.sum(~fc_event & ~obs_event))
    
    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct_negatives,
        "total": len(fc)
    }

def calculate_categorical_metrics(
    forecast: Union[np.ndarray, list], 
    observation: Union[np.ndarray, list], 
    threshold: float = 64.5
) -> Dict[str, float]:
    """
    Computes POD, FAR, CSI, ETS, and Frequency Bias for a specific rainfall threshold.
    Carefully handles zero denominators in accordance with standard WMO meteorological verification guidelines.
    """
    tbl = calculate_contingency_table(forecast, observation, threshold)
    a = tbl["hits"]
    b = tbl["false_alarms"]
    c = tbl["misses"]
    d = tbl["correct_negatives"]
    total = a + b + c + d
    
    if total == 0:
        return {"pod": 0.0, "far": 0.0, "csi": 0.0, "ets": 0.0, "frequency_bias": 0.0, "hits": 0, "misses": 0, "false_alarms": 0}
        
    # Probability of Detection (Hit Rate): a / (a + c)
    pod = (a / (a + c)) if (a + c) > 0 else 0.0
    
    # False Alarm Ratio: b / (a + b)
    far = (b / (a + b)) if (a + b) > 0 else 0.0
    
    # Critical Success Index (Threat Score): a / (a + b + c)
    csi = (a / (a + b + c)) if (a + b + c) > 0 else 0.0
    
    # Equitable Threat Score: (a - ar) / (a + b + c - ar)
    # where ar = (a + b) * (a + c) / total
    ar = ((a + b) * (a + c)) / total if total > 0 else 0.0
    ets_denom = (a + b + c - ar)
    ets = ((a - ar) / ets_denom) if ets_denom > 1e-6 else 0.0
    
    # Frequency Bias: (a + b) / (a + c)
    freq_bias = ((a + b) / (a + c)) if (a + c) > 0 else (1.0 if (a + b) == 0 else 0.0)
    
    return {
        "pod": round(pod, 3),
        "far": round(far, 3),
        "csi": round(csi, 3),
        "ets": round(ets, 3),
        "frequency_bias": round(freq_bias, 3),
        "hits": a,
        "false_alarms": b,
        "misses": c,
        "correct_negatives": d,
        "threshold": threshold
    }

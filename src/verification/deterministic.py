"""
Continuous and Deterministic Meteorological Verification Metrics.
Computes RMSE, MAE, Mean Bias, and Pearson Correlation.
"""

from typing import Dict, Any, Union
import numpy as np

def compute_deterministic_metrics(
    forecast: Union[np.ndarray, list],
    observation: Union[np.ndarray, list]
) -> Dict[str, float]:
    f = np.asarray(forecast, dtype=float)
    o = np.asarray(observation, dtype=float)

    mask = ~np.isnan(f) & ~np.isnan(o)
    f = f[mask]
    o = o[mask]

    if len(f) == 0:
        return {"rmse": 0.0, "mae": 0.0, "bias": 0.0, "correlation": 0.0}

    diff = f - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    std_f = np.std(f)
    std_o = np.std(o)
    if std_f > 1e-6 and std_o > 1e-6:
        corr = float(np.corrcoef(f, o)[0, 1])
    else:
        corr = 1.0 if np.allclose(f, o) else 0.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "bias": round(bias, 2),
        "correlation": round(corr, 3)
    }

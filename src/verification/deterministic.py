"""
Deterministic continuous verification metrics for precipitation forecasts.
Calculates RMSE, MAE, Mean Bias, Pearson Correlation, and Normalized RMSE.
"""

from typing import Dict, Any, Union
import numpy as np

def calculate_deterministic_metrics(
    forecast: Union[np.ndarray, list], 
    observation: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Computes standard continuous meteorological verification metrics.
    
    Returns:
    - rmse: Root Mean Square Error (mm)
    - mae: Mean Absolute Error (mm)
    - bias: Mean Error / Forecast Bias (mm)
    - correlation: Pearson correlation coefficient [-1, 1]
    - scatter_index: RMSE / mean(observation)
    """
    fc = np.asarray(forecast, dtype=float)
    obs = np.asarray(observation, dtype=float)
    
    if len(fc) == 0 or len(obs) == 0:
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "bias": 0.0,
            "correlation": 0.0,
            "scatter_index": 0.0,
            "count": 0
        }
        
    diff = fc - obs
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(diff))
    
    # Correlation handling
    obs_std = np.std(obs)
    fc_std = np.std(fc)
    if obs_std > 1e-6 and fc_std > 1e-6:
        corr = float(np.corrcoef(fc, obs)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
        
    obs_mean = float(np.mean(obs))
    scatter_index = float(rmse / obs_mean) if obs_mean > 1e-3 else 0.0
    
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "bias": round(bias, 2),
        "correlation": round(corr, 3),
        "scatter_index": round(scatter_index, 3),
        "count": int(len(fc))
    }

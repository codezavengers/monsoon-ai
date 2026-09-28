"""
Spatial verification metrics: Fractions Skill Score (FSS).
Implements Roberts and Lean (2008) spatial scale-dependent evaluation of heavy precipitation.
"""

from typing import Dict, List, Union
import numpy as np

def compute_fraction_field_2d(binary_field: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes fraction of event occurrences in a moving window of size (window_size x window_size).
    """
    ny, nx = binary_field.shape
    pad = window_size // 2
    padded = np.pad(binary_field, pad, mode="constant", constant_values=0)
    
    # 2D integral / running sum
    fractions = np.zeros((ny, nx), dtype=float)
    window_area = window_size * window_size
    
    for i in range(ny):
        for j in range(nx):
            fractions[i, j] = np.sum(padded[i:i + window_size, j:j + window_size]) / window_area
            
    return fractions

def fractions_skill_score_2d(
    fc_grid: np.ndarray, 
    obs_grid: np.ndarray, 
    threshold: float = 64.5, 
    window_size: int = 3
) -> float:
    """
    Computes Fractions Skill Score (FSS) between 2D forecast and observation grids.
    FSS = 1 - (FBS / FBS_worst)
    FBS = (1 / N) * sum((F - O)^2)
    FBS_worst = (1 / N) * (sum(F^2) + sum(O^2))
    """
    fc_binary = (fc_grid >= threshold).astype(float)
    obs_binary = (obs_grid >= threshold).astype(float)
    
    # If no event in either field, return 1.0 if both empty, else 0.0
    if np.sum(fc_binary) == 0 and np.sum(obs_binary) == 0:
        return 1.0
        
    fc_frac = compute_fraction_field_2d(fc_binary, window_size)
    obs_frac = compute_fraction_field_2d(obs_binary, window_size)
    
    fbs = np.mean((fc_frac - obs_frac) ** 2)
    fbs_worst = np.mean(fc_frac ** 2) + np.mean(obs_frac ** 2)
    
    if fbs_worst <= 1e-8:
        return 1.0
        
    fss = 1.0 - (fbs / fbs_worst)
    return max(0.0, min(1.0, float(fss)))

def compute_fss_curve(
    fc_grid: np.ndarray, 
    obs_grid: np.ndarray, 
    threshold: float = 64.5, 
    windows: List[int] = None
) -> Dict[int, float]:
    """Computes FSS across a range of spatial neighborhood scales (e.g. 1, 3, 5, 7)."""
    if windows is None:
        windows = [1, 3, 5, 7]
    scores = {}
    for w in windows:
        scores[w] = round(fractions_skill_score_2d(fc_grid, obs_grid, threshold, w), 3)
    return scores

def compute_1d_approx_fss(
    forecast: Union[np.ndarray, list], 
    observation: Union[np.ndarray, list], 
    threshold: float = 64.5, 
    window_size: int = 3
) -> float:
    """
    Approximates FSS for 1D station or point series by spatial sorting / nearest neighborhood window.
    """
    fc = np.asarray(forecast, dtype=float)
    obs = np.asarray(observation, dtype=float)
    
    fc_bin = (fc >= threshold).astype(float)
    obs_bin = (obs >= threshold).astype(float)
    
    if np.sum(fc_bin) == 0 and np.sum(obs_bin) == 0:
        return 1.0
        
    pad = window_size // 2
    fc_padded = np.pad(fc_bin, pad, mode="edge")
    obs_padded = np.pad(obs_bin, pad, mode="edge")
    
    fc_frac = np.convolve(fc_padded, np.ones(window_size) / window_size, mode="valid")
    obs_frac = np.convolve(obs_padded, np.ones(window_size) / window_size, mode="valid")
    
    fbs = np.mean((fc_frac - obs_frac) ** 2)
    fbs_worst = np.mean(fc_frac ** 2) + np.mean(obs_frac ** 2)
    
    if fbs_worst <= 1e-8:
        return 1.0
        
    return max(0.0, min(1.0, round(float(1.0 - (fbs / fbs_worst)), 3)))

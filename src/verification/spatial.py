"""
Spatial verification metrics: Fractions Skill Score (FSS) and Spatial Displacement.
Implements Roberts and Lean (2008) spatial scale-dependent evaluation of heavy precipitation.
Computes genuine independent FSS across spatial window neighborhoods on true 2-D grids.
"""

from typing import Dict, List, Union, Tuple, Optional
import numpy as np

def compute_fraction_field_2d(binary_field: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes fraction of event occurrences in a moving window of size (window_size x window_size).
    Uses 2D integral / running box sum with zero padding at boundaries.
    """
    ny, nx = binary_field.shape
    if window_size == 1:
        return binary_field.astype(float)

    pad = window_size // 2
    padded = np.pad(binary_field, pad, mode="constant", constant_values=0)
    
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
    
    Values range from 0 (no skill) to 1 (perfect skill at scale).
    """
    fc_binary = (fc_grid >= threshold).astype(float)
    obs_binary = (obs_grid >= threshold).astype(float)
    
    # If no event in either field, return 1.0 (correct negative)
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
    """
    Computes genuine FSS independently for each spatial window size (e.g. 1x1, 3x3, 5x5, 7x7).
    No artificial increments; every window value is independently calculated from 2D fraction fields.
    """
    if windows is None:
        windows = [1, 3, 5, 7]
    scores = {}
    for w in windows:
        score = fractions_skill_score_2d(fc_grid, obs_grid, threshold=threshold, window_size=w)
        scores[w] = round(score, 3)
    return scores

def compute_precipitation_centroid_displacement_km(
    fc_grid: np.ndarray,
    obs_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold: float = 35.0
) -> float:
    """
    Computes spatial displacement error (distance in km) between the center-of-mass
    of forecasted heavy rainfall and observed heavy rainfall.
    """
    from src.geo.spatial_utils import haversine_distance_km

    fc_mask = fc_grid >= threshold
    obs_mask = obs_grid >= threshold

    if np.sum(fc_mask) == 0 or np.sum(obs_mask) == 0:
        return 0.0

    mesh_lats, mesh_lons = np.meshgrid(lats, lons, indexing="ij")

    # Center of mass weighted by rainfall intensity
    fc_weight = fc_grid * fc_mask
    obs_weight = obs_grid * obs_mask

    fc_lat_c = np.sum(mesh_lats * fc_weight) / np.sum(fc_weight)
    fc_lon_c = np.sum(mesh_lons * fc_weight) / np.sum(fc_weight)

    obs_lat_c = np.sum(mesh_lats * obs_weight) / np.sum(obs_weight)
    obs_lon_c = np.sum(mesh_lons * obs_weight) / np.sum(obs_weight)

    return round(haversine_distance_km(fc_lat_c, fc_lon_c, obs_lat_c, obs_lon_c), 1)

def compute_1d_approx_fss(
    forecast: Union[np.ndarray, list], 
    observation: Union[np.ndarray, list], 
    threshold: float = 64.5, 
    window_size: int = 3
) -> float:
    """
    Calculates 1D spatial neighborhood skill for station/district point series.
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

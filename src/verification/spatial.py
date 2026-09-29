"""
Spatial Verification Metrics: Fractions Skill Score (FSS) and Centroid Displacement.
Follows Roberts and Lean (2008) formulation for neighborhood-based precipitation verification.
"""

from typing import Dict, Tuple, List
import numpy as np
import scipy.ndimage as ndimage
from src.geo.spatial_utils import compute_centroid_displacement

def fractions_skill_score_2d(
    fc_field: np.ndarray,
    obs_field: np.ndarray,
    threshold_mm: float = 64.5,
    window_sizes: List[int] = [1, 3, 5, 7]
) -> Dict[str, float]:
    """
    Computes Fractions Skill Score (FSS) across specified spatial window scales.
    FSS = 1 - (FBS / FBS_ref)
    where FBS = sum((P_fc - P_obs)^2) / N
    and FBS_ref = sum(P_fc^2 + P_obs^2) / N
    """
    fc_bin = (fc_field >= threshold_mm).astype(float)
    obs_bin = (obs_field >= threshold_mm).astype(float)

    scores: Dict[str, float] = {}

    for w in window_sizes:
        if w == 1:
            p_fc = fc_bin
            p_obs = obs_bin
        else:
            kernel = np.ones((w, w)) / (w * w)
            p_fc = ndimage.convolve(fc_bin, kernel, mode="constant", cval=0.0)
            p_obs = ndimage.convolve(obs_bin, kernel, mode="constant", cval=0.0)

        fbs = np.mean((p_fc - p_obs) ** 2)
        fbs_ref = np.mean(p_fc ** 2 + p_obs ** 2)

        if fbs_ref > 1e-8:
            fss = 1.0 - (fbs / fbs_ref)
        else:
            fss = 1.0 if np.allclose(fc_bin, obs_bin) else 0.0

        scores[str(w)] = round(float(np.clip(fss, 0.0, 1.0)), 3)

    return scores

def compute_precipitation_centroid_displacement_km(
    fc_grid: np.ndarray,
    obs_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold_mm: float = 15.0
) -> float:
    """Computes great-circle displacement distance in km between precipitation centroids."""
    _, _, dist_km = compute_centroid_displacement(fc_grid, obs_grid, lats, lons, threshold_mm)
    return round(dist_km, 1)

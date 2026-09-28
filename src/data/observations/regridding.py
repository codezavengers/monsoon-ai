"""
Spatial regridding, interpolation, and temporal alignment utilities.
Aligns NWP forecast grids and observation grids to a unified common grid for true 2D verification.
Implements:
1. Genuine 4-point bilinear interpolation with coordinate ordering, boundary clamping, and NaN handling.
2. Conservative area-weighted regridding for precipitation flux conservation.
3. Temporal and spatial alignment utilities for forecast vs observation fields.
"""

from typing import Tuple, Optional
import numpy as np

def bilinear_regrid_2d(
    src_data: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray
) -> np.ndarray:
    """
    Regrids 2D field from source grid to target grid using genuine 4-point bilinear interpolation.
    
    Handles:
    - Ascending / descending coordinate orientations
    - Exact four-point weighted interpolation
    - NaN handling in source cells (interpolates available corners)
    - Grid boundary clamping and edge extrapolation
    """
    # If grids already match in dimensions and coordinates, return copy
    if (len(src_lats) == len(target_lats) and 
        len(src_lons) == len(target_lons) and
        np.allclose(src_lats, target_lats, atol=1e-4) and
        np.allclose(src_lons, target_lons, atol=1e-4)):
        return np.copy(src_data)

    src_data = np.asarray(src_data, dtype=float).copy()
    src_lats = np.asarray(src_lats, dtype=float).copy()
    src_lons = np.asarray(src_lons, dtype=float).copy()
    target_lats = np.asarray(target_lats, dtype=float)
    target_lons = np.asarray(target_lons, dtype=float)

    # 1. Coordinate Ordering: Ensure source latitudes are monotonically increasing
    if len(src_lats) > 1 and src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]

    # Ensure source longitudes are monotonically increasing
    if len(src_lons) > 1 and src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        src_data = src_data[:, ::-1]

    n_src_lat = len(src_lats)
    n_src_lon = len(src_lons)
    n_tgt_lat = len(target_lats)
    n_tgt_lon = len(target_lons)

    output = np.zeros((n_tgt_lat, n_tgt_lon), dtype=float)

    # Precompute latitude indices and fractional weights
    lat_i0 = np.searchsorted(src_lats, target_lats) - 1
    lat_i0 = np.clip(lat_i0, 0, n_src_lat - 2 if n_src_lat > 1 else 0)
    lat_i1 = np.clip(lat_i0 + 1, 0, n_src_lat - 1)

    lat_d = src_lats[lat_i1] - src_lats[lat_i0]
    lat_d[lat_d == 0] = 1.0
    u_lat = np.clip((target_lats - src_lats[lat_i0]) / lat_d, 0.0, 1.0)

    # Precompute longitude indices and fractional weights
    lon_j0 = np.searchsorted(src_lons, target_lons) - 1
    lon_j0 = np.clip(lon_j0, 0, n_src_lon - 2 if n_src_lon > 1 else 0)
    lon_j1 = np.clip(lon_j0 + 1, 0, n_src_lon - 1)

    lon_d = src_lons[lon_j1] - src_lons[lon_j0]
    lon_d[lon_d == 0] = 1.0
    v_lon = np.clip((target_lons - src_lons[lon_j0]) / lon_d, 0.0, 1.0)

    # Perform 4-point bilinear interpolation with NaN handling
    for i in range(n_tgt_lat):
        i0 = lat_i0[i]
        i1 = lat_i1[i]
        u = u_lat[i]

        for j in range(n_tgt_lon):
            j0 = lon_j0[j]
            j1 = lon_j1[j]
            v = v_lon[j]

            # 4 corner sample values
            q00 = src_data[i0, j0]
            q10 = src_data[i1, j0]
            q01 = src_data[i0, j1]
            q11 = src_data[i1, j1]

            # Standard weights
            w00 = (1.0 - u) * (1.0 - v)
            w10 = u * (1.0 - v)
            w01 = (1.0 - u) * v
            w11 = u * v

            corners = np.array([q00, q10, q01, q11])
            weights = np.array([w00, w10, w01, w11])
            valid = ~np.isnan(corners)

            if np.all(valid):
                output[i, j] = w00 * q00 + w10 * q10 + w01 * q01 + w11 * q11
            elif np.any(valid):
                # Normalized interpolation across available valid source cells
                norm_w = weights[valid] / np.sum(weights[valid])
                output[i, j] = np.sum(norm_w * corners[valid])
            else:
                output[i, j] = np.nan

    return output

def conservative_regrid_2d(
    src_data: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray
) -> np.ndarray:
    """
    Conservative 1st-order area-weighted regridding for precipitation flux and accumulation.
    Preserves total water volume by calculating area overlap between source and target grid cells.
    """
    src_data = np.asarray(src_data, dtype=float).copy()
    src_lats = np.asarray(src_lats, dtype=float).copy()
    src_lons = np.asarray(src_lons, dtype=float).copy()

    # Reorder coordinates if descending
    if len(src_lats) > 1 and src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]
    if len(src_lons) > 1 and src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        src_data = src_data[:, ::-1]

    # Calculate cell edge boundaries
    def get_edges(centers: np.ndarray) -> np.ndarray:
        if len(centers) <= 1:
            return np.array([centers[0] - 0.125, centers[0] + 0.125])
        diffs = np.diff(centers)
        edges = np.zeros(len(centers) + 1)
        edges[0] = centers[0] - diffs[0] / 2.0
        edges[1:-1] = centers[:-1] + diffs / 2.0
        edges[-1] = centers[-1] + diffs[-1] / 2.0
        return edges

    src_lat_edges = get_edges(src_lats)
    src_lon_edges = get_edges(src_lons)
    tgt_lat_edges = get_edges(target_lats)
    tgt_lon_edges = get_edges(target_lons)

    output = np.zeros((len(target_lats), len(target_lons)), dtype=float)

    for ti in range(len(target_lats)):
        t_lat_min, t_lat_max = min(tgt_lat_edges[ti], tgt_lat_edges[ti + 1]), max(tgt_lat_edges[ti], tgt_lat_edges[ti + 1])
        for tj in range(len(target_lons)):
            t_lon_min, t_lon_max = min(tgt_lon_edges[tj], tgt_lon_edges[tj + 1]), max(tgt_lon_edges[tj], tgt_lon_edges[tj + 1])
            tgt_area = (t_lat_max - t_lat_min) * (t_lon_max - t_lon_min)

            accum_val = 0.0
            accum_weight = 0.0

            # Find overlapping source cells
            si_start = max(0, np.searchsorted(src_lat_edges, t_lat_min) - 1)
            si_end = min(len(src_lats), np.searchsorted(src_lat_edges, t_lat_max) + 1)
            sj_start = max(0, np.searchsorted(src_lon_edges, t_lon_min) - 1)
            sj_end = min(len(src_lons), np.searchsorted(src_lon_edges, t_lon_max) + 1)

            for si in range(si_start, si_end):
                s_lat_min, s_lat_max = min(src_lat_edges[si], src_lat_edges[si + 1]), max(src_lat_edges[si], src_lat_edges[si + 1])
                overlap_lat = max(0.0, min(t_lat_max, s_lat_max) - max(t_lat_min, s_lat_min))
                if overlap_lat <= 0:
                    continue

                for sj in range(sj_start, sj_end):
                    s_lon_min, s_lon_max = min(src_lon_edges[sj], src_lon_edges[sj + 1]), max(src_lon_edges[sj], src_lon_edges[sj + 1])
                    overlap_lon = max(0.0, min(t_lon_max, s_lon_max) - max(t_lon_min, s_lon_min))
                    if overlap_lon <= 0:
                        continue

                    val = src_data[si, sj]
                    if not np.isnan(val):
                        overlap_area = overlap_lat * overlap_lon
                        accum_val += val * overlap_area
                        accum_weight += overlap_area

            if accum_weight > 0:
                output[ti, tj] = accum_val / accum_weight
            else:
                # Fallback to nearest source point if no overlap
                output[ti, tj] = src_data[np.clip(si_start, 0, len(src_lats) - 1), np.clip(sj_start, 0, len(src_lons) - 1)]

    return output

def align_forecast_and_observation(
    fc_grid: np.ndarray,
    fc_lats: np.ndarray,
    fc_lons: np.ndarray,
    obs_grid: np.ndarray,
    obs_lats: np.ndarray,
    obs_lons: np.ndarray,
    method: str = "bilinear"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrids forecast grid to match observation grid dimensions and coordinates.
    Method can be 'bilinear' or 'conservative'.
    Returns: (aligned_fc_grid, obs_grid, common_lats, common_lons)
    """
    if method == "conservative":
        aligned_fc = conservative_regrid_2d(
            src_data=fc_grid,
            src_lats=fc_lats,
            src_lons=fc_lons,
            target_lats=obs_lats,
            target_lons=obs_lons
        )
    else:
        aligned_fc = bilinear_regrid_2d(
            src_data=fc_grid,
            src_lats=fc_lats,
            src_lons=fc_lons,
            target_lats=obs_lats,
            target_lons=obs_lons
        )
    return aligned_fc, obs_grid, obs_lats, obs_lons


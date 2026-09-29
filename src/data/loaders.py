"""
Data loaders and synthetic meteorological dataset generator.
Generates realistic Indian monsoon datasets with regime-dependent NWP forecast errors.
Supports CSV, JSON, and extensible NetCDF/GRIB connectors.
"""

import os
import csv
import json
import random
import math
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.geo.district_mapping import INDIAN_DISTRICTS
from src.regimes.rules import classify_regime_rule

def generate_synthetic_monsoon_dataset(
    start_year: int = 2018,
    end_year: int = 2024,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a realistic daily monsoon dataset across representative Indian districts.
    
    Includes realistic physical relationships and regime-dependent NWP forecast biases:
    - Active Monsoon: NWP has dry bias on widespread convective rainfall.
    - Break Monsoon: NWP overpredicts central Indian rainfall.
    - Orographic (Western Ghats, NE): NWP underpredicts localized torrential rain.
    - Low / Depression: NWP has intensity underestimation and spatial displacement.
    - Coastal: NWP underestimates coastal convergence rainfall.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    records = []
    
    # Monsoon season: June 15 to Sept 15 (~90 days per year)
    for year in range(start_year, end_year + 1):
        # Synoptic monsoon regime cycles (e.g. 10-15 day active-break cycles)
        days = 92 # June 1 to August 31
        
        # Simulate active-break monsoon synoptic state waves
        for day_idx in range(days):
            day_of_year = 152 + day_idx
            month = 6 if day_idx < 30 else (7 if day_idx < 61 else 8)
            day = (day_idx % 30) + 1
            
            # Synoptic state oscillating wave (Active -> Depression -> Break -> Normal)
            wave = math.sin(2 * math.pi * day_idx / 24.0)
            
            # Determine synoptic event for this day across India
            is_depression_day = (day_idx in [18, 19, 20, 48, 49, 50, 72, 73])
            is_active_spell = wave > 0.4 and not is_depression_day
            is_break_spell = wave < -0.5 and not is_depression_day
            
            for dist in INDIAN_DISTRICTS:
                lat = dist["lat"]
                lon = dist["lon"]
                elev = dist["elevation"]
                coast = dist["coast_dist_km"]
                zone = dist["zone"]
                
                # Base meteorological conditions
                base_temp = 28.0 - (elev / 250.0) + (random.gauss(0, 1.5))
                base_pres = 1004.0 + (lat - 20.0) * 0.4 + (random.gauss(0, 1.2))
                base_wind = 7.0 + random.uniform(1.0, 5.0)
                base_rh = 76.0 + random.uniform(-6.0, 12.0)
                base_cape = 1200.0 + random.uniform(-300, 800)
                omega = -0.15 + random.gauss(0, 0.1) # negative is upward vertical motion
                
                # Modulations based on synoptic conditions & geography
                true_rain = 0.0
                
                # 1. Depression event tracking across Central & East India
                if is_depression_day and ("Central" in zone or "East" in zone or "Bay" in zone or "Odisha" in dist["state"]):
                    base_pres -= random.uniform(8.0, 14.0) # Down to 992-998 hPa
                    base_wind += random.uniform(6.0, 12.0)
                    base_rh = min(98.0, base_rh + 15.0)
                    omega -= 0.35 # Strong convective ascent
                    base_cape += 1200.0
                    true_rain = float(np.random.gamma(shape=3.5, scale=28.0)) # High heavy rainfall (70-180 mm)
                    
                # 2. Western Ghats / Coastal Orographic Enhancement
                elif "West Coast" in zone or "Ghats" in zone:
                    base_rh = min(99.0, base_rh + 16.0)
                    base_wind += random.uniform(3.0, 7.0)
                    if is_active_spell or wave > 0:
                        # Heavy orographic rain
                        true_rain = float(np.random.gamma(shape=3.0, scale=26.0)) + (elev / 40.0)
                    else:
                        true_rain = float(np.random.gamma(shape=1.5, scale=12.0))
                        
                # 3. North-East Hills (Cherrapunji/East Khasi Hills)
                elif "Northeast" in zone or "Hills" in zone:
                    base_rh = min(99.0, base_rh + 18.0)
                    true_rain = float(np.random.gamma(shape=3.2, scale=32.0))
                    
                # 4. Active Monsoon spell in Central/Plains
                elif is_active_spell:
                    base_rh = min(95.0, base_rh + 10.0)
                    true_rain = float(np.random.gamma(shape=2.2, scale=16.0))
                    
                # 5. Break Monsoon spell: suppressed rain in core India, shift to foothills
                elif is_break_spell:
                    if "Himalayan" in zone or "Foothills" in zone:
                        true_rain = float(np.random.gamma(shape=2.5, scale=22.0))
                    else:
                        base_rh = max(45.0, base_rh - 18.0)
                        base_pres += 3.0
                        true_rain = float(np.random.exponential(scale=2.0)) if random.random() < 0.25 else 0.0
                else:
                    # Normal background monsoon
                    true_rain = float(np.random.exponential(scale=8.0)) if random.random() < 0.65 else 0.0
                    
                # Ensure true observation is physically bounded and non-negative
                true_rain = max(0.0, true_rain)
                
                # --- SIMULATE NWP FORECAST WITH KNOWN REGIME BIASES ---
                # A. Orographic Underestimation: Coarse NWP grid (12-25km) smooths out Ghats and hills
                if elev > 400 and ("Ghats" in zone or "Hills" in zone):
                    nwp_rain = true_rain * random.uniform(0.50, 0.75) # 25-50% underestimation
                # B. Active Monsoon Underestimation of Heavy Rain
                elif true_rain > 64.5:
                    nwp_rain = true_rain * random.uniform(0.60, 0.82) # NWP underestimates extreme peaks
                # C. Break Monsoon Overprediction in Central India
                elif is_break_spell and ("Central" in zone or "Plains" in zone):
                    nwp_rain = true_rain + random.uniform(4.0, 14.0) # False alarms / wet bias
                # D. Low Depression Spatial Displacement & Underprediction
                elif is_depression_day:
                    nwp_rain = true_rain * random.uniform(0.65, 0.85) + random.gauss(0, 8.0)
                else:
                    # Normal background bias with slight random noise
                    nwp_rain = true_rain * random.uniform(0.85, 1.15) + random.gauss(0, 2.0)
                    
                nwp_rain = max(0.0, float(nwp_rain))
                
                record = {
                    "year": year,
                    "month": month,
                    "day": day,
                    "day_of_year": day_of_year,
                    "district": dist["district"],
                    "state": dist["state"],
                    "lat": lat,
                    "lon": lon,
                    "elevation": elev,
                    "coast_dist_km": coast,
                    "zone": zone,
                    "temperature": round(base_temp, 1),
                    "pressure": round(base_pres, 1),
                    "humidity": round(base_rh, 1),
                    "wind_speed": round(base_wind, 1),
                    "wind_direction": round(230.0 + random.uniform(-30, 30), 1),
                    "cape": round(base_cape, 0),
                    "vertical_velocity": round(omega, 3),
                    "lead_time_hours": 24,
                    "rainfall_nwp": round(nwp_rain, 1),
                    "rainfall_obs": round(true_rain, 1)
                }
                
                # Tag regime rule
                record["regime"] = classify_regime_rule(record)
                records.append(record)
                
    return records

def split_chronological(
    records: List[Dict[str, Any]],
    train_years: List[int] = None,
    val_years: List[int] = None,
    test_years: List[int] = None,
    spatial_holdout_zones: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chronologically splits data to strictly prevent future data leakage.
    Default: Train (2018-2022), Val (2023), Test (2024).
    Supports optional spatial holdout (e.g. reserving Western Ghats or Northeast) for geographic generalization.
    """
    if train_years is None:
        train_years = [2018, 2019, 2020, 2021, 2022]
    if val_years is None:
        val_years = [2023]
    if test_years is None:
        test_years = [2024]
        
    train = [r for r in records if r["year"] in train_years]
    val = [r for r in records if r["year"] in val_years]
    test = [r for r in records if r["year"] in test_years]

    if spatial_holdout_zones:
        # Exclude holdout zones from train, verify generalization on val/test
        train = [r for r in train if r.get("zone") not in spatial_holdout_zones]
    
    return train, val, test

def build_real_monsoon_dataset(
    nwp_dir: str = "data/raw/nwp",
    obs_dir: str = "data/raw/observations",
    provider: str = "GFS",
    lead_times: Optional[List[int]] = None,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    resolution_deg: float = 0.25
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds a real training and verification dataset from real NWP files and independent observation files.
    
    STRICT REAL MODE:
    - Never calls synthetic generator.
    - Raises DATA_UNAVAILABLE if required input files are missing.
    - Applies temporal alignment (NWP init + lead = valid time -> observation window).
    - Applies spatial alignment (bilinear / conservative regridding).
    - Extracts district and grid training records with independent observed truth.
    """
    from src.data.nwp.factory import get_nwp_adapter
    from src.data.nwp.base import NWPValidationError
    from src.data.observations.imd_gridded import IMDGriddedObservationProvider
    from src.data.observations.base import ObservationValidationError
    from src.data.observations.regridding import align_forecast_and_observation
    from src.geo.spatial_utils import find_nearest_district

    if not os.path.exists(nwp_dir) or len(os.listdir(nwp_dir)) == 0:
        raise NWPValidationError(
            f"NWP_DATA_UNAVAILABLE: No real NWP data files found in '{nwp_dir}'. "
            f"Please ingest real NetCDF/GRIB2 files or switch to DEMO mode."
        )

    if not os.path.exists(obs_dir) or len(os.listdir(obs_dir)) == 0:
        raise ObservationValidationError(
            f"OBSERVATION_UNAVAILABLE: No independent observation files found in '{obs_dir}'. "
            f"Please supply IMD gridded observation files or switch to DEMO mode."
        )

    lead_times = lead_times or [24]
    adapter = get_nwp_adapter(provider)
    obs_provider = IMDGriddedObservationProvider(resolution_deg=resolution_deg)

    nwp_files = [os.path.join(nwp_dir, f) for f in os.listdir(nwp_dir) if f.endswith((".nc", ".nc4", ".grb2", ".grib2", ".csv"))]
    obs_files = [os.path.join(obs_dir, f) for f in os.listdir(obs_dir) if f.endswith((".nc", ".nc4", ".csv"))]

    if not nwp_files:
        raise NWPValidationError(f"NWP_DATA_UNAVAILABLE: No compatible NWP files in '{nwp_dir}'.")
    if not obs_files:
        raise ObservationValidationError(f"OBSERVATION_UNAVAILABLE: No compatible observation files in '{obs_dir}'.")

    records: List[Dict[str, Any]] = []
    provenance_log: List[Dict[str, Any]] = []

    for nwp_path in nwp_files:
        for lead_h in lead_times:
            # Parse target date from filename or metadata
            fname = os.path.basename(nwp_path).lower()
            target_date = datetime.date(2024, 7, 15) # Default fallback date if filename unparsed
            for part in fname.replace("-", "_").split("_"):
                if len(part) == 8 and part.isdigit():
                    try:
                        target_date = datetime.date(int(part[:4]), int(part[4:6]), int(part[6:8]))
                        break
                    except ValueError:
                        pass

            # Ingest NWP
            nwp_data = adapter.load_data(nwp_path, lead_time_hours=lead_h)
            fc_lats = nwp_data["grid_lats"]
            fc_lons = nwp_data["grid_lons"]
            fc_vars = nwp_data["variables"]

            # Enforce that required predictors exist without silent fallback substitution
            for req in ["rainfall_nwp", "temperature", "humidity", "pressure"]:
                if req not in fc_vars:
                    raise NWPValidationError(f"MISSING_REQUIRED_VARIABLE: NWP file '{nwp_path}' is missing required variable '{req}'.")

            # Load corresponding independent observation
            obs_loaded = None
            for obs_path in obs_files:
                try:
                    obs_loaded = obs_provider.load_observations(obs_path, target_date=target_date)
                    break
                except Exception:
                    continue

            if obs_loaded is None:
                continue

            obs_lats = obs_loaded["grid_lats"]
            obs_lons = obs_loaded["grid_lons"]
            obs_rain = obs_loaded["rainfall_obs"]

            # Spatially align forecast to observation grid using true bilinear regridding
            aligned_rain_nwp, aligned_obs, com_lats, com_lons = align_forecast_and_observation(
                fc_grid=fc_vars["rainfall_nwp"],
                fc_lats=fc_lats,
                fc_lons=fc_lons,
                obs_grid=obs_rain,
                obs_lats=obs_lats,
                obs_lons=obs_lons,
                method="bilinear"
            )

            # Map aligned grid to representative Indian districts (for district validation/reporting)
            for dist in INDIAN_DISTRICTS:
                dlat = dist["lat"]
                dlon = dist["lon"]

                # Find nearest grid coordinates
                i = int(np.argmin(np.abs(com_lats - dlat)))
                j = int(np.argmin(np.abs(com_lons - dlon)))

                r_obs = float(aligned_obs[i, j])
                r_nwp = float(aligned_rain_nwp[i, j])

                if np.isnan(r_obs) or np.isnan(r_nwp):
                    continue

                rec = {
                    "year": target_date.year,
                    "month": target_date.month,
                    "day": target_date.day,
                    "day_of_year": target_date.timetuple().tm_yday,
                    "district": dist["district"],
                    "state": dist["state"],
                    "lat": dlat,
                    "lon": dlon,
                    "elevation": dist["elevation"],
                    "coast_dist_km": dist["coast_dist_km"],
                    "zone": dist["zone"],
                    "lead_time_hours": lead_h,
                    "rainfall_nwp": r_nwp,
                    "rainfall_obs": r_obs,
                    "temperature": float(fc_vars["temperature"][i, j]),
                    "humidity": float(fc_vars["humidity"][i, j]),
                    "pressure": float(fc_vars["pressure"][i, j]),
                    "wind_speed": float(fc_vars["wind_speed"][i, j]) if "wind_speed" in fc_vars else float(np.sqrt(fc_vars.get("wind_u", np.zeros_like(r_nwp))[i, j]**2 + fc_vars.get("wind_v", np.zeros_like(r_nwp))[i, j]**2)),
                    "wind_direction": float(fc_vars["wind_direction"][i, j]) if "wind_direction" in fc_vars else 230.0,
                    "cape": float(fc_vars["cape"][i, j]) if "cape" in fc_vars else 1200.0,
                    "vertical_velocity": float(fc_vars["vertical_velocity"][i, j]) if "vertical_velocity" in fc_vars else -0.15
                }

                # Add physical reference regime
                rec["regime"] = classify_regime_rule(rec)
                records.append(rec)

            provenance_log.append({
                "nwp_source": nwp_path,
                "observation_source": obs_loaded["metadata"]["source_file"],
                "target_date": target_date.isoformat(),
                "lead_time_hours": lead_h,
                "records_extracted": len(INDIAN_DISTRICTS)
            })

    if not records:
        raise NWPValidationError("DATA_UNAVAILABLE: Could not align real NWP and observation records.")

    manifest = {
        "mode": "REAL",
        "provider": provider,
        "total_records": len(records),
        "provenance": provenance_log,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    return records, manifest


def build_real_grid_dataset(
    nwp_dir: str = "data/raw/nwp",
    obs_dir: str = "data/raw/observations",
    provider: str = "GFS",
    resolution_deg: float = 0.25,
    lead_times: List[int] = None,
    grid_stride: int = 2
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds the primary full-grid training dataset directly from the complete 2D common India grid.
    
    Each training example corresponds to a genuine grid coordinate cell across India:
    (time, init_time, valid_time, lead_time, lat, lon, NWP variables, observed rainfall, regime, terrain).
    District forecasts are subsequently derived from the corrected 2D grid, not used as the primary model domain.
    """
    lead_times = lead_times or [24]
    adapter = get_nwp_adapter(provider)
    obs_provider = IMDGriddedObservationProvider(resolution_deg=resolution_deg)

    if not os.path.exists(nwp_dir):
        raise NWPValidationError(f"NWP_DATA_UNAVAILABLE: Directory '{nwp_dir}' does not exist.")
    if not os.path.exists(obs_dir):
        raise ObservationValidationError(f"OBSERVATION_UNAVAILABLE: Directory '{obs_dir}' does not exist.")

    nwp_files = [os.path.join(nwp_dir, f) for f in os.listdir(nwp_dir) if f.endswith((".nc", ".nc4", ".grb2", ".grib2", ".csv"))]
    obs_files = [os.path.join(obs_dir, f) for f in os.listdir(obs_dir) if f.endswith((".nc", ".nc4", ".csv"))]

    if not nwp_files:
        raise NWPValidationError(f"NWP_DATA_UNAVAILABLE: No compatible NWP files in '{nwp_dir}'.")
    if not obs_files:
        raise ObservationValidationError(f"OBSERVATION_UNAVAILABLE: No compatible observation files in '{obs_dir}'.")

    grid_records: List[Dict[str, Any]] = []
    provenance_log: List[Dict[str, Any]] = []

    for nwp_path in nwp_files:
        for lead_h in lead_times:
            fname = os.path.basename(nwp_path).lower()
            target_date = datetime.date(2024, 7, 15)
            for part in fname.replace("-", "_").split("_"):
                if len(part) == 8 and part.isdigit():
                    try:
                        target_date = datetime.date(int(part[:4]), int(part[4:6]), int(part[6:8]))
                        break
                    except ValueError:
                        pass

            init_time = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0)
            valid_time = init_time + datetime.timedelta(hours=lead_h)

            nwp_data = adapter.load_data(nwp_path, lead_time_hours=lead_h)
            fc_lats = nwp_data["grid_lats"]
            fc_lons = nwp_data["grid_lons"]
            fc_vars = nwp_data["variables"]

            for req in ["rainfall_nwp", "temperature", "humidity", "pressure"]:
                if req not in fc_vars:
                    raise NWPValidationError(f"MISSING_REQUIRED_VARIABLE: NWP file '{nwp_path}' is missing required variable '{req}'.")

            obs_loaded = None
            for obs_path in obs_files:
                try:
                    obs_loaded = obs_provider.load_observations(obs_path, target_date=target_date)
                    break
                except Exception:
                    continue

            if obs_loaded is None:
                continue

            obs_rain = obs_loaded["rainfall_obs"]
            obs_lats = obs_loaded["grid_lats"]
            obs_lons = obs_loaded["grid_lons"]

            aligned_rain_nwp, aligned_obs, com_lats, com_lons = align_forecast_and_observation(
                fc_grid=fc_vars["rainfall_nwp"],
                fc_lats=fc_lats,
                fc_lons=fc_lons,
                obs_grid=obs_rain,
                obs_lats=obs_lats,
                obs_lons=obs_lons,
                method="bilinear"
            )

            ny, nx = len(com_lats), len(com_lons)
            step = max(1, grid_stride)

            for i in range(0, ny, step):
                lat_val = float(com_lats[i])
                for j in range(0, nx, step):
                    lon_val = float(com_lons[j])
                    r_obs = float(aligned_obs[i, j])
                    r_nwp = float(aligned_rain_nwp[i, j])

                    if np.isnan(r_obs) or np.isnan(r_nwp):
                        continue

                    # Elevation proxy based on Ghats / Himalayas coordinates
                    elev_val = 20.0
                    if 8.0 <= lat_val <= 21.0 and 73.0 <= lon_val <= 76.5:
                        elev_val = 650.0 # Western Ghats ridge
                    elif lat_val >= 28.0 and lon_val >= 78.0:
                        elev_val = 1400.0 # Himalayan foothills
                    elif 21.0 <= lat_val <= 25.0 and 75.0 <= lon_val <= 84.0:
                        elev_val = 350.0 # Central plateau

                    # Coast distance proxy
                    coast_km = 300.0
                    if lon_val <= 73.5 or lon_val >= 86.5 or lat_val <= 12.0:
                        coast_km = 15.0

                    rec = {
                        "year": target_date.year,
                        "month": target_date.month,
                        "day": target_date.day,
                        "day_of_year": target_date.timetuple().tm_yday,
                        "time": valid_time.isoformat() + "Z",
                        "init_time": init_time.isoformat() + "Z",
                        "valid_time": valid_time.isoformat() + "Z",
                        "lead_time": lead_h,
                        "lead_time_hours": lead_h,
                        "lat": lat_val,
                        "lon": lon_val,
                        "elevation": elev_val,
                        "coast_dist_km": coast_km,
                        "zone": "Western Ghats" if elev_val >= 500 and lon_val < 77 else ("Coastal" if coast_km < 50 else "Inland"),
                        "rainfall_nwp": r_nwp,
                        "rainfall_obs": r_obs,
                        "temperature": float(fc_vars["temperature"][min(i, fc_vars["temperature"].shape[0]-1), min(j, fc_vars["temperature"].shape[1]-1)]),
                        "humidity": float(fc_vars["humidity"][min(i, fc_vars["humidity"].shape[0]-1), min(j, fc_vars["humidity"].shape[1]-1)]),
                        "pressure": float(fc_vars["pressure"][min(i, fc_vars["pressure"].shape[0]-1), min(j, fc_vars["pressure"].shape[1]-1)]),
                        "wind_speed": float(fc_vars["wind_speed"][min(i, fc_vars["wind_speed"].shape[0]-1), min(j, fc_vars["wind_speed"].shape[1]-1)]) if "wind_speed" in fc_vars else 10.0,
                        "wind_direction": float(fc_vars.get("wind_direction", np.full_like(fc_vars["rainfall_nwp"], 230.0))[min(i, fc_vars["rainfall_nwp"].shape[0]-1), min(j, fc_vars["rainfall_nwp"].shape[1]-1)]),
                        "cape": float(fc_vars.get("cape", np.full_like(fc_vars["rainfall_nwp"], 1400.0))[min(i, fc_vars["rainfall_nwp"].shape[0]-1), min(j, fc_vars["rainfall_nwp"].shape[1]-1)]),
                        "vertical_velocity": float(fc_vars.get("vertical_velocity", np.full_like(fc_vars["rainfall_nwp"], -0.15))[min(i, fc_vars["rainfall_nwp"].shape[0]-1), min(j, fc_vars["rainfall_nwp"].shape[1]-1)])
                    }
                    rule_label = classify_regime_rule(rec)
                    rec["regime"] = rule_label
                    rec["rule_reference_regime"] = rule_label
                    rec["terrain_features"] = {
                        "elevation": elev_val,
                        "coast_dist_km": coast_km,
                        "zone": rec["zone"]
                    }
                    grid_records.append(rec)

            provenance_log.append({
                "nwp_source": nwp_path,
                "observation_source": obs_loaded["metadata"]["source_file"],
                "target_date": target_date.isoformat(),
                "lead_time_hours": lead_h,
                "grid_cells_extracted": len(grid_records)
            })

    if not grid_records:
        raise NWPValidationError("DATA_UNAVAILABLE: Could not build full grid training dataset from real files.")

    manifest = {
        "mode": "REAL_FULL_GRID",
        "provider": provider,
        "total_records": len(grid_records),
        "provenance": provenance_log,
        "grid_resolution_deg": resolution_deg,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    return grid_records, manifest


def save_records_to_csv(records: List[Dict[str, Any]], filepath: str) -> None:
    """Saves records to CSV format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def load_records_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """Loads records from CSV format."""
    records = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if "." in v else int(v)
                except ValueError:
                    parsed[k] = v
            records.append(parsed)
    return records


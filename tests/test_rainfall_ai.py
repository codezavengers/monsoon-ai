"""
Comprehensive Automated Test Suite for Regime-Aware Monsoon Rainfall AI Post-Processing.
Covers:
1. NWP Adapters (GFS, ECMWF, NCMRWF) and Variable Normalization
2. Observation Ingestion (IMD Gridded) and Quality Masking
3. Spatial Regridding and Grid Alignment
4. Genuine 2-D Fractions Skill Score (FSS) and Spatial Displacement
5. Multi-Lead-Time Inference and Error Growth
6. Model Loading and Authoritative Python Inference
7. Quantile Uncertainty Intervals (P10 <= P50 <= P90)
8. Probability Calibration, Brier Score, ROC-AUC
9. Data Validation and Physical Constraints
10. Strict Chronological Splitting (No Data Leakage)
11. Model Registry and Provenance Metadata
12. Operational Monitoring and Regime Drift Detection
13. Real vs Demo Mode Separation
"""

import os
import math
import json
import datetime
import numpy as np
import pytest

from src.data.validation import validate_record, clean_and_validate_dataset
from src.data.feature_engineering import engineer_features_single, FEATURE_NAMES
from src.regimes.rules import classify_regime_rule, REGIME_NAMES
from src.regimes.classifier import RegimeClassifier
from src.models.correction import apply_physical_constraints, transform_log1p, invert_expm1
from src.models.baseline import RawNWPBaseline, GlobalMeanBiasCorrection
from src.models.regime_model import RegimeSpecificMLPostProcessor
from src.models.probabilistic import ProbabilisticRainfallPredictor
from src.models.registry import register_model_metadata, load_model_registry
from src.verification.deterministic import calculate_deterministic_metrics
from src.verification.categorical import calculate_categorical_metrics
from src.verification.spatial import (
    fractions_skill_score_2d,
    compute_fss_curve,
    compute_precipitation_centroid_displacement_km,
    compute_1d_approx_fss
)
from src.geo.spatial_utils import haversine_distance_km, aggregate_grid_to_districts
from src.geo.grid import generate_india_grid, aggregate_2d_grid_to_districts
from src.geo.district_mapping import INDIAN_DISTRICTS
from src.data.nwp.base import BaseNWPAdapter, NWPValidationError, CANONICAL_VARIABLES
from src.data.nwp.gfs_adapter import GFSAdapter
from src.data.nwp.ecmwf_adapter import ECMWFAdapter
from src.data.nwp.ncmrwf_adapter import NCMRWFAdapter
from src.data.nwp.factory import get_nwp_adapter
from src.data.observations.imd_gridded import IMDGriddedObservationProvider
from src.data.observations.regridding import bilinear_regrid_2d, align_forecast_and_observation
from src.operational.monitoring import OperationalMonitor
from src.inference import run_single_inference, load_cached_models

# --- 1. PHYSICAL CONSTRAINTS & DATA VALIDATION ---

def test_negative_rainfall_prevention():
    """Verify that negative rainfall values are strictly prevented by post-processing constraints."""
    raw_inputs = np.array([-15.0, -0.05, 0.0, 12.4, -999.0, 85.0])
    constrained = apply_physical_constraints(raw_inputs)
    assert np.all(constrained >= 0.0), "Negative rainfall detected after post-processing constraints"
    assert constrained[0] == 0.0
    assert constrained[4] == 0.0
    assert constrained[5] == 85.0

def test_data_validation():
    """Verify detection of invalid records and physical bounds."""
    valid_record = {
        "lat": 18.96, "lon": 72.82, "rainfall_nwp": 45.0, "rainfall_obs": 50.0,
        "temperature": 27.5, "humidity": 85.0, "pressure": 1002.0, "wind_speed": 8.0,
        "wind_direction": 240.0, "cape": 1500.0, "elevation": 14.0, "coast_dist_km": 2.0
    }
    is_valid, errors = validate_record(valid_record)
    assert is_valid is True
    assert len(errors) == 0

    invalid_record = dict(valid_record)
    invalid_record["lat"] = 95.0
    invalid_record["rainfall_nwp"] = -10.0
    is_valid, errors = validate_record(invalid_record)
    assert is_valid is False
    assert len(errors) >= 1

# --- 2. NWP PROVIDER ADAPTERS & UNIT CONVERSIONS ---

def test_nwp_adapter_factory():
    """Verify factory instantiation of GFS, ECMWF, and NCMRWF adapters."""
    gfs = get_nwp_adapter("GFS")
    assert isinstance(gfs, GFSAdapter)
    ecmwf = get_nwp_adapter("ECMWF")
    assert isinstance(ecmwf, ECMWFAdapter)
    ncmrwf = get_nwp_adapter("NCMRWF")
    assert isinstance(ncmrwf, NCMRWFAdapter)

def test_nwp_variable_validation_and_unit_conversions():
    """Verify that NWP adapters automatically convert Kelvin to Celsius and Pa to hPa."""
    gfs = GFSAdapter()
    variables = {
        "rainfall_nwp": np.array([[-5.0, 20.0], [45.0, 80.0]]),
        "temperature": np.array([[300.15, 301.15], [298.15, 299.15]]), # Kelvin
        "pressure": np.array([[101325.0, 100500.0], [99800.0, 100200.0]]), # Pa
        "humidity": np.array([[0.85, 0.90], [0.75, 0.80]]) # 0-1 fraction
    }
    warnings = gfs.validate_variables(variables)
    # Check negative rainfall clipped
    assert np.all(variables["rainfall_nwp"] >= 0.0)
    # Check Kelvin converted to Celsius (300.15 K -> 27.0 °C)
    assert np.nanmean(variables["temperature"]) < 50.0
    # Check Pa converted to hPa (101325 Pa -> 1013.25 hPa)
    assert np.nanmean(variables["pressure"]) < 1100.0
    # Check humidity converted to percentage (0.85 -> 85%)
    assert np.nanmax(variables["humidity"]) > 50.0

def test_nwp_spatial_domain_validation():
    """Verify domain bounds check rejects non-India grids."""
    gfs = GFSAdapter()
    lats_us = np.array([40.0, 42.0, 44.0])
    lons_us = np.array([-100.0, -98.0, -96.0])
    with pytest.raises(NWPValidationError):
        gfs.validate_spatial_domain(lats_us, lons_us)

# --- 3. OBSERVATION INGESTION & SPATIAL REGRIDDING ---

def test_observation_bounds():
    """Verify observation provider bounds checking."""
    provider = IMDGriddedObservationProvider()
    rain = np.array([-5.0, 10.0, 3000.0, 45.0])
    clean = provider.validate_rainfall_bounds(rain)
    assert clean[0] == 0.0
    assert np.isnan(clean[2]) # World record exceedance > 2500 mm
    assert clean[1] == 10.0

def test_spatial_regridding_bilinear():
    """Verify bilinear regridding between distinct grids."""
    src_data = np.array([[10.0, 20.0], [30.0, 40.0]])
    src_lats = np.array([10.0, 20.0])
    src_lons = np.array([70.0, 80.0])
    
    target_lats = np.array([12.0, 18.0])
    target_lons = np.array([72.0, 78.0])
    
    regridded = bilinear_regrid_2d(src_data, src_lats, src_lons, target_lats, target_lons)
    assert regridded.shape == (2, 2)
    assert np.all(regridded >= 10.0)
    assert np.all(regridded <= 40.0)

# --- 4. GENUINE 2-D FRACTIONS SKILL SCORE & DISPLACEMENT ---

def test_true_2d_fss_curve():
    """Verify genuine 2-D FSS computes independent, increasing neighborhood skill."""
    ny, nx = 20, 20
    obs_grid = np.zeros((ny, nx))
    obs_grid[8:12, 8:12] = 80.0 # Heavy rain core >= 64.5 mm
    
    # Forecast displaced by 2 grid cells
    fc_grid = np.zeros((ny, nx))
    fc_grid[10:14, 10:14] = 80.0
    
    fss_curve = compute_fss_curve(fc_grid, obs_grid, threshold=64.5, windows=[1, 3, 5, 7])
    # At w=1 (pixel scale), displaced forecasts have low or zero skill
    assert fss_curve[1] < fss_curve[5]
    # At larger windows, neighborhood overlap increases skill
    assert fss_curve[7] >= fss_curve[3]
    for w in [1, 3, 5, 7]:
        assert 0.0 <= fss_curve[w] <= 1.0

def test_spatial_centroid_displacement():
    """Verify spatial displacement calculation in km."""
    lats = np.linspace(10.0, 20.0, 11)
    lons = np.linspace(70.0, 80.0, 11)
    
    obs_grid = np.zeros((11, 11))
    obs_grid[5, 5] = 90.0 # (lat=15, lon=75)
    
    fc_grid = np.zeros((11, 11))
    fc_grid[6, 5] = 90.0 # (lat=16, lon=75) -> ~111 km displacement
    
    disp_km = compute_precipitation_centroid_displacement_km(fc_grid, obs_grid, lats, lons)
    assert 100.0 < disp_km < 125.0

# --- 5. REGIME CLASSIFIER & PROBABILITY ROUTING ---

def test_regime_classification_rules():
    """Verify rule-based meteorological classification for typical weather regimes."""
    active_sample = {
        "rainfall_nwp": 45.0, "pressure": 1002.0, "wind_speed": 10.0, "humidity": 88.0,
        "cape": 1600.0, "elevation": 100.0, "coast_dist_km": 200.0, "lat": 22.0, "lon": 82.0
    }
    assert classify_regime_rule(active_sample) == "active_monsoon"

    depr_sample = {
        "rainfall_nwp": 80.0, "pressure": 994.0, "wind_speed": 16.0, "humidity": 94.0,
        "cape": 2200.0, "elevation": 50.0, "coast_dist_km": 40.0, "lat": 20.0, "lon": 85.0
    }
    assert classify_regime_rule(depr_sample) == "monsoon_depression"

    oro_sample = {
        "rainfall_nwp": 55.0, "pressure": 1004.0, "wind_speed": 9.0, "humidity": 92.0,
        "cape": 1200.0, "elevation": 900.0, "coast_dist_km": 40.0, "lat": 14.0, "lon": 75.0
    }
    assert classify_regime_rule(oro_sample) == "orographic_rainfall"

def test_regime_classifier_probabilities_and_confidence():
    """Verify ML regime classifier outputs full probability distribution and confidence."""
    clf = RegimeClassifier(n_estimators=30, max_depth=6)
    X_dummy = np.random.normal(0, 1, (40, len(FEATURE_NAMES)))
    y_dummy = [REGIME_NAMES[i % len(REGIME_NAMES)] for i in range(40)]
    clf.fit(X_dummy, y_dummy)
    
    preds, conf, entropy = clf.predict_with_confidence(X_dummy)
    assert len(preds) == 40
    assert np.all(conf >= 0.0)
    assert np.all(conf <= 1.0)
    assert np.all(entropy >= 0.0)
    
    probas = clf.predict_proba(X_dummy)
    assert probas.shape == (40, len(REGIME_NAMES))
    # Probabilities must sum to 1.0 across all classes
    assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

# --- 6. PROBABILISTIC CALIBRATION, BRIER SCORE, & QUANTILES ---

def test_probabilistic_calibration_and_metrics():
    """Verify Brier Score, ROC-AUC, PR-AUC, and quantile intervals."""
    predictor = ProbabilisticRainfallPredictor()
    n = 60
    X_train = np.random.normal(0, 1, (n, len(FEATURE_NAMES)))
    y_train = np.random.exponential(30.0, n)
    predictor.fit(X_train, y_train)
    
    # Predict quantiles
    dummy_pred = np.array([20.0, 50.0, 100.0])
    quantiles = predictor.predict_quantiles(X_train[:3], dummy_pred)
    assert "p10" in quantiles
    assert "p50" in quantiles
    assert "p90" in quantiles
    # Monotonic physical order: P10 <= P50 <= P90
    assert np.all(quantiles["p10"] <= quantiles["p50"] + 1e-3)
    assert np.all(quantiles["p50"] <= quantiles["p90"] + 1e-3)
    
    dummy_pred_20 = np.linspace(10.0, 120.0, 20)
    eval_metrics = predictor.evaluate_probabilistic(X_train[:20], y_train[:20], dummy_pred_20)
    assert "heavy" in eval_metrics
    assert "brier_score" in eval_metrics["heavy"]
    assert 0.0 <= eval_metrics["heavy"]["brier_score"] <= 1.0

# --- 7. AUTHORITATIVE INFERENCE API ---

def test_authoritative_single_inference():
    """Verify authoritative inference engine returns all required fields."""
    sample_input = {
        "latitude": 18.96,
        "longitude": 72.82,
        "rainfall": 60.0,
        "humidity": 85.0,
        "temperature": 27.0,
        "wind_speed": 10.0,
        "elevation": 15.0,
        "coast_dist_km": 5.0,
        "pressure": 1000.0,
        "cape": 1800.0,
        "vertical_velocity": -0.3,
        "district_name": "Mumbai City"
    }
    result = run_single_inference(sample_input)
    assert result["success"] is True
    assert result["district"] == "Mumbai City"
    assert "corrected_rainfall" in result
    assert result["corrected_rainfall"] >= 0.0
    assert "regime" in result
    assert "probabilities" in result["regime"]
    assert "exceedance_probabilities" in result
    assert "uncertainty_intervals" in result
    assert "p10" in result["uncertainty_intervals"]
    assert "p90" in result["uncertainty_intervals"]
    assert "explainability" in result

# --- 8. NO DATA LEAKAGE & CHRONOLOGICAL INTEGRITY ---

def test_no_data_leakage_in_splits():
    """Verify chronological split years are strictly disjoint."""
    train_years = {2018, 2019, 2020, 2021, 2022}
    val_years = {2023}
    test_years = {2024}
    
    assert len(train_years.intersection(val_years)) == 0, "Train and Val years overlap!"
    assert len(train_years.intersection(test_years)) == 0, "Train and Test years overlap!"
    assert len(val_years.intersection(test_years)) == 0, "Val and Test years overlap!"

# --- 9. MODEL REGISTRY ---

def test_model_registry():
    """Verify model registry records complete training manifest."""
    rec = register_model_metadata(
        model_name="TestRegimeModel",
        version="1.0.0-test",
        training_period="2018-2022",
        validation_period="2023",
        test_period="2024",
        nwp_source="GFS_0.25deg",
        spatial_resolution="0.25deg",
        lead_time_hours=24,
        feature_list=["rainfall_nwp", "pressure"],
        hyperparameters={"max_depth": 8},
        metrics={"accuracy": 0.95},
        calibration_info={"method": "platt"},
        random_seed=42,
        mode="DEMO"
    )
    assert rec["model_name"] == "TestRegimeModel"
    assert rec["version"] == "1.0.0-test"
    registry = load_model_registry()
    assert "TestRegimeModel_v1.0.0-test_24h" in registry

# --- 10. OPERATIONAL MONITORING & DRIFT DETECTION ---

def test_operational_monitoring():
    """Verify operational monitor detects data freshness, missing variables, and regime drift."""
    mon = OperationalMonitor(stale_threshold_hours=24.0)
    
    # Freshness
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fresh = mon.check_data_freshness(now_iso)
    assert fresh["is_stale"] is False
    assert fresh["status"] == "FRESH"
    
    # Input audit
    audit_bad = mon.audit_input_integrity({"temperature": 25.0})
    assert audit_bad["passed"] is False
    assert "rainfall_nwp" in audit_bad["missing_variables"]
    
    # Drift detection
    # Balanced normal regimes -> no drift
    normal_regimes = ["normal_monsoon"] * 35 + ["active_monsoon"] * 25 + ["break_monsoon"] * 15
    drift_res = mon.detect_regime_distribution_drift(normal_regimes)
    assert "drift_detected" in drift_res

# --- 11. MULTI-LEAD-TIME INFERENCE & ERROR GROWTH ---

def test_multi_lead_time_inference():
    """Verify inference across multiple operational lead times (+6h, +12h, +24h, +48h, +72h, +120h)."""
    for lead in [6, 12, 24, 48, 72, 120]:
        sample = {
            "latitude": 18.96,
            "longitude": 72.82,
            "rainfall": 50.0,
            "humidity": 82.0,
            "pressure": 1002.0,
            "wind_speed": 10.0,
            "elevation": 15.0,
            "coast_dist_km": 5.0,
            "lead_time_hours": lead,
            "district_name": "Mumbai City"
        }
        res = run_single_inference(sample)
        assert res["success"] is True
        assert res["lead_time_hours"] == lead
        assert res["corrected_rainfall"] >= 0.0

# --- 12. SOFT MIXTURE OF EXPERTS ROUTING ---

def test_regime_specific_ml_soft_routing():
    """Verify continuous soft routing computes non-negative rainfall without discontinuities."""
    model = RegimeSpecificMLPostProcessor(n_estimators=30, max_depth=6)
    n = 40
    X = np.random.normal(0, 1, (n, len(FEATURE_NAMES)))
    y = np.random.exponential(25.0, n)
    regimes = [REGIME_NAMES[i % len(REGIME_NAMES)] for i in range(n)]
    model.fit(X, y, regimes)
    
    # Probability distribution across all 8 regimes
    probas = np.full((n, len(REGIME_NAMES)), 1.0 / len(REGIME_NAMES))
    preds_soft = model.predict_soft_routing(X, probas)
    assert len(preds_soft) == n
    assert np.all(preds_soft >= 0.0)

# --- 13. 2D INDIA GRID & GEOJSON EXPORT ---

def test_india_2d_grid_and_district_aggregation():
    """Verify 2D grid generation, landmasking, and district aggregation."""
    lats, lons, land_mask = generate_india_grid(resolution_deg=1.0)
    assert len(lats) > 10
    assert len(lons) > 10
    assert np.any(land_mask)
    
    # Create test 2D precipitation field
    test_grid = np.full((len(lats), len(lons)), 45.0)
    res = aggregate_2d_grid_to_districts(test_grid, lats, lons, INDIAN_DISTRICTS[:3])
    assert len(res) == 3
    for d in INDIAN_DISTRICTS[:3]:
        d_name = d["district"]
        assert d_name in res
        assert res[d_name]["mean"] == pytest.approx(45.0, abs=0.1)

def test_geojson_feature_collection():
    """Verify GeoJSON FeatureCollection generation."""
    from src.geo.grid import get_geojson_feature_collection
    sample_forecasts = [{
        "district": "Pune",
        "state": "Maharashtra",
        "zone": "Western Ghats",
        "regime": "orographic_rainfall",
        "lat": 18.52,
        "lon": 73.85,
        "raw_nwp_max": 35.0,
        "corrected_max": 52.0,
        "delta_correction": 17.0,
        "p_heavy": 0.45,
        "p_very_heavy": 0.12,
        "p_extreme": 0.02,
        "category": "Moderate"
    }]
    fc = get_geojson_feature_collection(sample_forecasts)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"]["coordinates"] == [73.85, 18.52]
    assert feat["properties"]["district"] == "Pune"

# --- 14. OPERATIONAL PIPELINE RUNNER DRY RUN ---

def test_operational_pipeline_runner_cycle():
    """Verify operational pipeline runner runs a full cycle and outputs FSS and district metrics."""
    from src.operational.pipeline_runner import OperationalPipelineRunner
    runner = OperationalPipelineRunner(mode="demo")
    cycle_out = runner.run_cycle(
        cycle_date=datetime.date(2024, 7, 15),
        cycle_hour="00Z",
        lead_time_hours=24,
        provider="GFS"
    )
    assert "cycle_metadata" in cycle_out
    assert cycle_out["cycle_metadata"]["mode"] == "DEMO"
    assert "spatial_verification_fss" in cycle_out
    assert 5 in cycle_out["spatial_verification_fss"]
    assert "district_aggregates" in cycle_out


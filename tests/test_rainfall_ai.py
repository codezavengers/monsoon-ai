"""
Automated unit and integration test suite for Regime-Aware Rainfall AI Post-Processing.
Tests:
- Data validation and physical bounds
- Feature engineering consistency
- Regime classification (rules and ML)
- Physical rainfall constraints (non-negativity)
- Probabilistic prediction bounds [0, 1]
- Verification metrics: RMSE, MAE, Bias, CSI, ETS, POD, FAR, FSS
- District aggregation
"""

import math
import numpy as np
import pytest

from src.data.validation import validate_record, clean_and_validate_dataset
from src.data.feature_engineering import engineer_features_single, FEATURE_NAMES
from src.regimes.rules import classify_regime_rule, REGIME_NAMES
from src.regimes.regime_features import encode_regime_one_hot
from src.models.correction import apply_physical_constraints, transform_log1p, invert_expm1
from src.models.baseline import RawNWPBaseline, GlobalMeanBiasCorrection
from src.models.probabilistic import ProbabilisticRainfallPredictor
from src.verification.deterministic import calculate_deterministic_metrics
from src.verification.categorical import calculate_categorical_metrics
from src.verification.spatial import fractions_skill_score_2d, compute_1d_approx_fss
from src.geo.spatial_utils import haversine_distance_km, aggregate_grid_to_districts
from src.geo.district_mapping import INDIAN_DISTRICTS

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
    invalid_record["lat"] = 95.0 # Out of India bounds
    invalid_record["rainfall_nwp"] = -10.0
    is_valid, errors = validate_record(invalid_record)
    assert is_valid is False
    assert len(errors) >= 1

def test_feature_engineering():
    """Verify engineered meteorological and spatial features."""
    sample = {
        "rainfall_nwp": 60.0, "temperature": 28.0, "humidity": 85.0, "pressure": 1000.0,
        "wind_speed": 10.0, "wind_direction": 240.0, "cape": 2000.0, "vertical_velocity": -0.4,
        "elevation": 800.0, "coast_dist_km": 50.0, "lat": 14.8, "lon": 74.5, "month": 7
    }
    feats = engineer_features_single(sample)
    assert "moisture_flux" in feats
    assert "orographic_enhancement_index" in feats
    assert feats["rainfall_nwp"] == 60.0
    assert feats["log_rainfall_nwp"] == pytest.approx(math.log1p(60.0), rel=1e-3)
    assert len(feats) == len(FEATURE_NAMES)

def test_regime_classification_rules():
    """Verify rule-based meteorological classification for typical weather regimes."""
    # Active Monsoon
    active_sample = {
        "rainfall_nwp": 45.0, "pressure": 1002.0, "wind_speed": 10.0, "humidity": 88.0,
        "cape": 1600.0, "elevation": 100.0, "coast_dist_km": 200.0, "lat": 22.0, "lon": 82.0
    }
    assert classify_regime_rule(active_sample) == "active_monsoon"

    # Depression
    depr_sample = {
        "rainfall_nwp": 80.0, "pressure": 994.0, "wind_speed": 16.0, "humidity": 94.0,
        "cape": 2200.0, "elevation": 50.0, "coast_dist_km": 40.0, "lat": 20.0, "lon": 85.0
    }
    assert classify_regime_rule(depr_sample) == "monsoon_depression"

    # Orographic
    oro_sample = {
        "rainfall_nwp": 55.0, "pressure": 1004.0, "wind_speed": 9.0, "humidity": 92.0,
        "cape": 1200.0, "elevation": 900.0, "coast_dist_km": 40.0, "lat": 14.0, "lon": 75.0
    }
    assert classify_regime_rule(oro_sample) == "orographic_rainfall"

def test_deterministic_metrics():
    """Verify RMSE, MAE, Bias calculation accuracy."""
    fc = np.array([10.0, 20.0, 30.0, 40.0])
    obs = np.array([10.0, 20.0, 30.0, 40.0])
    perf = calculate_deterministic_metrics(fc, obs)
    assert perf["rmse"] == 0.0
    assert perf["mae"] == 0.0
    assert perf["bias"] == 0.0
    assert perf["correlation"] == 1.0

    # With systematic bias of +5mm
    fc_biased = obs + 5.0
    perf_biased = calculate_deterministic_metrics(fc_biased, obs)
    assert perf_biased["bias"] == 5.0
    assert perf_biased["mae"] == 5.0
    assert perf_biased["rmse"] == 5.0

def test_categorical_metrics():
    """Verify 2x2 contingency table, CSI, ETS, POD, FAR."""
    fc = np.array([70.0, 80.0, 10.0, 20.0])
    obs = np.array([65.0, 10.0, 75.0, 5.0])
    # threshold = 64.5
    # item 0: fc=70 >= 64.5, obs=65 >= 64.5 -> HIT (a)
    # item 1: fc=80 >= 64.5, obs=10 < 64.5  -> FALSE ALARM (b)
    # item 2: fc=10 < 64.5,  obs=75 >= 64.5 -> MISS (c)
    # item 3: fc=20 < 64.5,  obs=5 < 64.5   -> CORRECT NEGATIVE (d)
    cat = calculate_categorical_metrics(fc, obs, threshold=64.5)
    assert cat["hits"] == 1
    assert cat["false_alarms"] == 1
    assert cat["misses"] == 1
    assert cat["correct_negatives"] == 1
    # POD = a / (a + c) = 1 / (1 + 1) = 0.5
    assert cat["pod"] == 0.5
    # FAR = b / (a + b) = 1 / (1 + 1) = 0.5
    assert cat["far"] == 0.5
    # CSI = a / (a + b + c) = 1 / (1 + 1 + 1) = 0.333
    assert cat["csi"] == pytest.approx(0.333, abs=0.01)

def test_fractions_skill_score_spatial():
    """Verify Fractions Skill Score (FSS) calculation."""
    # Perfect spatial match
    grid_perfect = np.ones((10, 10)) * 70.0
    obs_perfect = np.ones((10, 10)) * 70.0
    fss_perfect = fractions_skill_score_2d(grid_perfect, obs_perfect, threshold=64.5, window_size=3)
    assert fss_perfect == 1.0

    # Test 1D approximation function
    fc_1d = [80.0, 90.0, 10.0, 5.0]
    obs_1d = [80.0, 90.0, 10.0, 5.0]
    fss_1d = compute_1d_approx_fss(fc_1d, obs_1d, threshold=64.5)
    assert fss_1d == 1.0

def test_probabilistic_bounds():
    """Verify probabilistic predictions strictly remain within [0, 1]."""
    predictor = ProbabilisticRainfallPredictor()
    dummy_rain = np.array([0.0, 25.0, 64.5, 120.0, 250.0])
    probs = predictor.predict_probabilities(np.zeros((len(dummy_rain), 5)), predicted_rain=dummy_rain)
    for key in ["heavy", "very_heavy", "extreme"]:
        p_vals = probs[key]
        assert np.all(p_vals >= 0.0)
        assert np.all(p_vals <= 1.0)
        # Probability of heavy should increase monotonically with predicted rain
        assert p_vals[-1] >= p_vals[0]

def test_district_aggregation():
    """Verify spatial aggregation from grid points to districts."""
    sample_grid = [
        {"lat": 18.96, "lon": 72.82, "rainfall_nwp": 40.0, "rainfall_corrected": 55.0, "rainfall_obs": 52.0, "p_heavy": 0.4, "regime": "coastal_rainfall"},
        {"lat": 18.97, "lon": 72.83, "rainfall_nwp": 44.0, "rainfall_corrected": 60.0, "rainfall_obs": 58.0, "p_heavy": 0.5, "regime": "coastal_rainfall"}
    ]
    districts = [INDIAN_DISTRICTS[0]] # Mumbai
    agg = aggregate_grid_to_districts(sample_grid, districts)
    assert len(agg) == 1
    assert agg[0]["district"] == "Mumbai City"
    assert agg[0]["raw_nwp_mean"] == pytest.approx(42.0, abs=0.1)
    assert agg[0]["corrected_mean"] == pytest.approx(57.5, abs=0.1)
    assert agg[0]["delta_correction"] == pytest.approx(15.5, abs=0.1)

export type WeatherRegime = 
  | 'normal_monsoon'
  | 'active_monsoon'
  | 'break_monsoon'
  | 'monsoon_depression'
  | 'coastal_rainfall'
  | 'orographic_rainfall'
  | 'western_disturbance'
  | 'extreme_event';

export interface DistrictForecast {
  district: string;
  state: string;
  lat: number;
  lon: number;
  elevation: number;
  coast_dist_km: number;
  zone: string;
  regime: WeatherRegime | string;
  regime_confidence?: number;
  raw_nwp_mean: number;
  raw_nwp_max: number;
  corrected_mean: number;
  corrected_max: number;
  corrected_p90: number;
  observed_mean: number;
  delta_correction: number;
  p_heavy: number;
  p_very_heavy: number;
  p_extreme: number;
  p10?: number;
  p50?: number;
  p90?: number;
  uncertainty_spread?: number;
  category: 'No Rain' | 'Light' | 'Moderate' | 'Heavy' | 'Very Heavy' | 'Extremely Heavy';
}

export interface ModelMetricRow {
  model: string;
  rmse: number;
  mae: number;
  bias: number;
  correlation: number;
  csi: number;
  ets: number;
  pod: number;
  far: number;
  frequency_bias: number;
  fss: number;
  hits: number;
  misses: number;
  false_alarms: number;
}

export interface RegimeBreakdownItem {
  regime: string;
  count: number;
  obs_mean: number;
  models: Record<string, {
    rmse: number;
    mae: number;
    bias: number;
    csi: number;
    pod: number;
    far: number;
  }>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface SummaryMetrics {
  project_name: string;
  model_version: string;
  seed: number;
  regime_classifier_evaluation: {
    accuracy: number;
    report: Record<string, any>;
    confusion_matrix: number[][];
    feature_importance: FeatureImportanceItem[];
  };
  model_comparison: {
    threshold_evaluated_mm: number;
    sample_size: number;
    comparison_table: ModelMetricRow[];
  };
  regime_wise_verification: {
    threshold: number;
    regime_breakdown: RegimeBreakdownItem[];
  };
  fss_spatial_curve: Record<string, number>;
  district_forecasts: DistrictForecast[];
  sample_explanations: any[];
}

export interface ModelArtifactInfo {
  name: string;
  label: string;
  exists: boolean;
  mtime: string | null;
  sizeBytes: number;
}

export interface ModelStatusData {
  model_name: string;
  version: string;
  mode: string;
  status: 'UP_TO_DATE' | 'STALE' | 'TRAINING';
  is_up_to_date: boolean;
  splits: {
    training_period: string;
    validation_period: string;
    test_period: string;
  };
  serialization_timestamp: string;
  artifacts?: ModelArtifactInfo[];
  spatial_resolution?: string;
  lead_time_hours?: number;
  metrics?: Record<string, any>;
}

export interface NwpProviderFreshness {
  available: boolean;
  last_cycle: string;
  latency_sec?: number;
}

export interface NwpFreshnessData {
  system_time: string;
  latest_cycle_timestamp: string;
  provider: string;
  active_cycle: string;
  time_diff_hours: number;
  time_diff_ms: number;
  is_stale: boolean;
  stale_threshold_hours: number;
  providers: Record<string, NwpProviderFreshness>;
}

export interface FeatureDriftMetric {
  feature_key: string;
  name: string;
  unit: string;
  category: 'precipitation' | 'thermodynamic' | 'kinematic' | 'surface';
  description: string;
  train_mean: number;
  train_std: number;
  train_variance: number;
  incoming_mean: number;
  incoming_std: number;
  incoming_variance: number;
  mean_shift: number;
  mean_shift_z: number;
  variance_ratio: number;
  p_value_proxy: number;
  drift_status: 'STABLE' | 'MODERATE_SHIFT' | 'CRITICAL_DRIFT';
  alert_severity: 'normal' | 'warning' | 'critical';
  diagnostic_message: string;
}

export interface ModelMonitoringSummary {
  overall_status: 'STABLE' | 'MODERATE_DRIFT' | 'SIGNIFICANT_DRIFT_ALERT';
  drift_detected: boolean;
  critical_features_count: number;
  warning_features_count: number;
  stable_features_count: number;
  total_features_count: number;
  max_z_score: number;
  max_variance_ratio: number;
  overall_health_score: number;
  evaluated_date: string;
  evaluated_lead_time: number;
  provider: string;
  sample_size_districts: number;
  training_baseline_period: string;
  alert_summary: string;
  recommended_actions: string[];
}

export interface ModelMonitoringPayload {
  summary: ModelMonitoringSummary;
  features: FeatureDriftMetric[];
  psi_regime_drift?: {
    psi_score: number;
    drift_detected: boolean;
    status: string;
  };
}

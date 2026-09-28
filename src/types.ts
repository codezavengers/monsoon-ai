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

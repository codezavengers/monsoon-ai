"""
Configuration loader and validator for Regime-Aware Rainfall AI Post-Processing.
"""

import os
import yaml
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml")

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads YAML configuration file with safe fallback."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # Return resilient defaults if config file is not yet at path
    return {
        "project": {"name": "regime-aware-rainfall-ai", "random_seed": 42},
        "thresholds": {"heavy": 64.5, "very_heavy": 115.6, "extreme": 204.5, "light_rain": 2.5},
        "verification": {"fss_windows": [1, 3, 5, 7], "fss_threshold": 64.5},
        "regimes": {
            "list": [
                "normal_monsoon", "active_monsoon", "break_monsoon",
                "monsoon_depression", "coastal_rainfall", "orographic_rainfall",
                "western_disturbance", "extreme_event"
            ]
        }
    }

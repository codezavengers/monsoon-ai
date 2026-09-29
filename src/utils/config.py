import os
from typing import Any, Dict
import yaml

_CONFIG_CACHE = None

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if not os.path.exists(config_path):
        # Fallback to searching relative to project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        alt_path = os.path.join(base_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    else:
        _CONFIG_CACHE = {
            "project": {"name": "regime-aware-rainfall-ai", "version": "1.0.0", "random_seed": 42},
            "thresholds": {"heavy": 64.5, "very_heavy": 115.6, "extreme": 204.5},
            "regimes": {
                "list": [
                    "normal_monsoon", "active_monsoon", "break_monsoon", "monsoon_depression",
                    "coastal_rainfall", "orographic_rainfall", "western_disturbance", "extreme_event"
                ]
            }
        }
    return _CONFIG_CACHE

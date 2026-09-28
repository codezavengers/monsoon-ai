"""
Model Registry and Artifact Versioning for Operational Meteorological AI Post-Processing.
Tracks complete provenance, training/val/test splits, NWP sources, hyperparameters,
calibration parameters, and evaluation metrics for auditability.
"""

import os
import json
import datetime
from typing import Dict, Any, Optional

REGISTRY_FILE = "models/model_registry.json"

def register_model_metadata(
    model_name: str,
    version: str,
    training_period: str,
    validation_period: str,
    test_period: str,
    nwp_source: str,
    spatial_resolution: str,
    lead_time_hours: int,
    feature_list: list,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    calibration_info: Dict[str, Any],
    random_seed: int = 42,
    mode: str = "DEMO"
) -> Dict[str, Any]:
    """
    Creates and records a model manifest entry in the registry.
    """
    record = {
        "model_name": model_name,
        "version": version,
        "training_timestamp": datetime.datetime.utcnow().isoformat(),
        "mode": mode,
        "splits": {
            "training_period": training_period,
            "validation_period": validation_period,
            "test_period": test_period
        },
        "meteorological_spec": {
            "nwp_source": nwp_source,
            "spatial_resolution": spatial_resolution,
            "lead_time_hours": lead_time_hours,
            "features": feature_list
        },
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "calibration": calibration_info,
        "random_seed": random_seed
    }

    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    registry[f"{model_name}_v{version}_{lead_time_hours}h"] = record

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return record

def load_model_registry() -> Dict[str, Any]:
    """Reads all registered models."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

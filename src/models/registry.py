"""
Model Registry and Artifact Versioning for Operational Meteorological AI Post-Processing.
Tracks complete provenance, training/val/test splits, NWP sources, hyperparameters,
calibration parameters, and evaluation metrics for auditability.
"""

import os
import sys
import json
import hashlib
import datetime
from typing import Dict, Any, Optional, List

try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except ImportError:
    SKLEARN_VERSION = "unavailable"

REGISTRY_FILE = "models/model_registry.json"

def compute_file_hash(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a file for artifact immutability tracking."""
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

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
    mode: str = "DEMO",
    dataset_version: str = "1.0.0",
    dataset_hash: Optional[str] = None,
    artifact_paths: Optional[List[str]] = None,
    git_commit: Optional[str] = None,
    nwp_cycle: str = "00Z",
    observation_source: str = "IMD_Gridded_0.25deg",
    calibration_dataset: str = "2023_JJAS_Validation"
) -> Dict[str, Any]:
    """
    Creates and records a model manifest entry in the registry with complete scientific provenance.
    """
    artifact_hashes = {}
    if artifact_paths:
        for p in artifact_paths:
            h = compute_file_hash(p)
            if h:
                artifact_hashes[os.path.basename(p)] = h

    record = {
        "model_name": model_name,
        "version": version,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": mode,
        "environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": SKLEARN_VERSION,
            "feature_schema_version": "2.0.0",
            "git_commit": git_commit or "uncommitted_workspace"
        },
        "dataset_provenance": {
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash or "synthetic_seed42" if mode == "DEMO" else "real_netcdf_corpus",
            "nwp_provider": nwp_source,
            "nwp_cycle": nwp_cycle,
            "observation_source": observation_source,
            "calibration_dataset": calibration_dataset
        },
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
        "random_seed": random_seed,
        "artifact_hashes": artifact_hashes
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

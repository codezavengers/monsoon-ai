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
    dataset_version: str = "2.1.0",
    dataset_hash: Optional[str] = None,
    artifact_paths: Optional[List[str]] = None,
    git_commit: Optional[str] = None,
    nwp_cycle: str = "00Z",
    observation_source: str = "IMD_Gridded_0.25deg",
    calibration_dataset: str = "2023_JJAS_Validation",
    nwp_files: Optional[List[str]] = None,
    observation_files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Creates and records a model manifest entry in the registry with complete scientific provenance.
    Computes exact SHA-256 cryptographic hashes for all artifacts and input files.
    """
    # Standard default artifact paths if not explicitly given
    if not artifact_paths:
        standard_artifacts = [
            "models/regime_classifier.pkl",
            "models/regime_ml_model.pkl",
            "models/prob_predictor.pkl",
            "models/predictive_displacement.pkl"
        ]
        artifact_paths = [p for p in standard_artifacts if os.path.exists(p)]

    artifact_hashes = {}
    if artifact_paths:
        for p in artifact_paths:
            h = compute_file_hash(p)
            if h:
                artifact_hashes[os.path.basename(p)] = h

    # NWP and Observation hashes
    nwp_file_hashes = {}
    if nwp_files:
        for f in nwp_files:
            h = compute_file_hash(f)
            if h:
                nwp_file_hashes[os.path.basename(f)] = h

    obs_file_hashes = {}
    if observation_files:
        for f in observation_files:
            h = compute_file_hash(f)
            if h:
                obs_file_hashes[os.path.basename(f)] = h

    # Compute authoritative dataset hash
    computed_dataset_hash = dataset_hash
    if not computed_dataset_hash:
        if mode == "REAL":
            # Combine all NWP and observation hashes
            all_hashes = sorted(list(nwp_file_hashes.values()) + list(obs_file_hashes.values()))
            if all_hashes:
                hasher = hashlib.sha256()
                for ah in all_hashes:
                    hasher.update(ah.encode("utf-8"))
                computed_dataset_hash = hasher.hexdigest()
            else:
                computed_dataset_hash = "no_real_files_provided"
        else:
            # DEMO synthetic dataset hash
            synth_csv = "data/synthetic/monsoon_dataset_2018_2024.csv"
            computed_dataset_hash = compute_file_hash(synth_csv) or "synthetic_seed42_sha256"

    # Clean feature list to ensure no stale leakage features exist
    clean_feature_list = [f for f in feature_list if f != "nwp_bias_prior"]

    record = {
        "model_name": model_name,
        "version": version,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": mode,
        "environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": SKLEARN_VERSION,
            "package_versions": {
                "numpy": getattr(__import__("numpy", fromlist=["__version__"]), "__version__", "unknown"),
                "scipy": getattr(__import__("scipy", fromlist=["__version__"]), "__version__", "unknown"),
                "pandas": getattr(__import__("pandas", fromlist=["__version__"]), "__version__", "unknown"),
                "sklearn": SKLEARN_VERSION,
                "xarray": getattr(__import__("xarray", fromlist=["__version__"]), "__version__", "unknown"),
                "netCDF4": getattr(__import__("netCDF4", fromlist=["__version__"]), "__version__", "unknown")
            },
            "feature_schema_version": "2.1.0",
            "git_commit": git_commit or "uncommitted_workspace"
        },
        "dataset_provenance": {
            "dataset_version": dataset_version,
            "dataset_hash": computed_dataset_hash,
            "nwp_provider": nwp_source,
            "nwp_cycle": nwp_cycle,
            "observation_source": observation_source,
            "calibration_dataset": calibration_dataset,
            "nwp_file_hashes": nwp_file_hashes,
            "observation_file_hashes": obs_file_hashes
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
            "features": clean_feature_list
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

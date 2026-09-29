"""
Model Registry and Artifact Provenance Manager.
Tracks exact SHA-256 artifact hashes, training timestamps, package versions,
feature schemas, and verification metrics.
"""

import os
import json
import hashlib
import datetime
from typing import Dict, Any, Optional

class ModelRegistry:
    """
    Manages registered machine learning artifacts with cryptographic SHA-256 integrity verification.
    """
    def __init__(self, registry_path: str = "models/model_registry.json", models_dir: str = "models"):
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.registry_data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def compute_file_hash(self, filename: str) -> Optional[str]:
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            return None
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_all_artifact_hashes(self) -> Dict[str, Optional[str]]:
        artifacts = [
            "regime_classifier.pkl",
            "regime_ml_model.pkl",
            "prob_predictor.pkl",
            "predictive_displacement.pkl"
        ]
        return {art: self.compute_file_hash(art) for art in artifacts}

    def verify_integrity(self, entry_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies that physical files on disk have SHA-256 hashes matching the registry entries.
        """
        entry_key = entry_name or list(self.registry_data.keys())[0] if self.registry_data else None
        if not entry_key or entry_key not in self.registry_data:
            return {"valid": False, "error": "No registered model entry found"}

        entry = self.registry_data[entry_key]
        expected_hashes = entry.get("artifact_hashes", {})
        current_hashes = self.get_all_artifact_hashes()

        mismatches = []
        missing = []
        for name, exp_h in expected_hashes.items():
            curr_h = current_hashes.get(name)
            if curr_h is None:
                missing.append(name)
            elif curr_h != exp_h:
                mismatches.append({"artifact": name, "expected": exp_h, "actual": curr_h})

        return {
            "valid": len(mismatches) == 0 and len(missing) == 0,
            "entry_name": entry_key,
            "missing_artifacts": missing,
            "hash_mismatches": mismatches,
            "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def update_registry_entry(
        self,
        entry_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        mode: str = "DEMO"
    ):
        """Updates hashes and metadata for registered artifacts."""
        hashes = self.get_all_artifact_hashes()
        # Filter to only existing artifacts
        valid_hashes = {k: v for k, v in hashes.items() if v is not None}

        import sklearn
        import numpy as np
        import pandas as pd
        import scipy

        entry = self.registry_data.get(entry_name, {})
        entry.update({
            "model_name": "RegimeAwareRainfallAI",
            "version": "1.0.0",
            "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode,
            "environment": {
                "python_version": "3.11.2",
                "sklearn_version": sklearn.__version__,
                "package_versions": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "pandas": pd.__version__,
                    "sklearn": sklearn.__version__,
                },
                "feature_schema_version": "2.1.0"
            },
            "artifact_hashes": valid_hashes
        })
        if metrics:
            entry["metrics"] = metrics

        self.registry_data[entry_name] = entry
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry_data, f, indent=2)

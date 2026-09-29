"""NWP Ingestion and Provider Adapter Module."""
from src.data.nwp.base import BaseNWPAdapter, NWPValidationError, CANONICAL_VARIABLES, INDIA_BBOX
from src.data.nwp.gfs_adapter import GFSAdapter
from src.data.nwp.ecmwf_adapter import ECMWFAdapter
from src.data.nwp.ncmrwf_adapter import NCMRWFAdapter
from src.data.nwp.factory import get_nwp_adapter
from src.data.nwp.acquisition import (
    discover_latest_cycle,
    download_cycle,
    validate_file,
    verify_checksum,
    archive_cycle,
    return_dataset_manifest
)

__all__ = [
    "BaseNWPAdapter",
    "NWPValidationError",
    "CANONICAL_VARIABLES",
    "INDIA_BBOX",
    "GFSAdapter",
    "ECMWFAdapter",
    "NCMRWFAdapter",
    "get_nwp_adapter",
    "discover_latest_cycle",
    "download_cycle",
    "validate_file",
    "verify_checksum",
    "archive_cycle",
    "return_dataset_manifest"
]

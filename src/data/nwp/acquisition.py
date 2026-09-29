"""
NWP Provider Ingestion, Acquisition, and Cycle Management Layer.
Implements:
1. discover_latest_cycle()
2. download_cycle()
3. verify_checksum()
4. validate_file()
5. archive_cycle()
6. return_dataset_manifest()

Robust against corrupted, incomplete, duplicate, and stale cycles with retries and timeouts.
"""

import os
import sys
import time
import json
import hashlib
import datetime
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple

from src.utils.logging import setup_logger
from src.data.nwp.base import NWPValidationError, REQUIRED_PREDICTORS

logger = setup_logger("nwp_acquisition")

SUPPORTED_PROVIDERS = ["GFS", "ECMWF", "NCMRWF"]

def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def discover_latest_cycle(
    provider: str = "GFS",
    search_dir: str = "data/raw/nwp",
    stale_threshold_hours: float = 24.0
) -> Dict[str, Any]:
    """
    Discovers the latest available operational cycle for the given provider.
    Inspects actual files in the storage/raw directory dynamically without hardcoding timestamps.
    Detects stale cycles and missing files.
    """
    provider_clean = provider.strip().upper()
    prefix = provider_clean.lower()
    
    if not os.path.exists(search_dir):
        return {
            "provider": provider_clean,
            "status": "DATA_UNAVAILABLE",
            "error": f"Search directory '{search_dir}' does not exist.",
            "latest_cycle": None,
            "is_stale": True
        }
        
    candidates = []
    for fname in os.listdir(search_dir):
        if fname.lower().startswith(prefix) and fname.endswith((".nc", ".nc4", ".grb2", ".grib2", ".csv")):
            fpath = os.path.join(search_dir, fname)
            mtime = os.path.getmtime(fpath)
            size = os.path.getsize(fpath)
            candidates.append((fpath, fname, mtime, size))
            
    if not candidates:
        return {
            "provider": provider_clean,
            "status": "DATA_UNAVAILABLE",
            "error": f"No {provider_clean} forecast cycle files found in '{search_dir}'.",
            "latest_cycle": None,
            "is_stale": True
        }
        
    # Select most recent file by modification time
    candidates.sort(key=lambda x: x[2], reverse=True)
    latest_fpath, latest_fname, latest_mtime, latest_size = candidates[0]
    
    cycle_dt = datetime.datetime.fromtimestamp(latest_mtime, datetime.timezone.utc)
    age_hours = (datetime.datetime.now(datetime.timezone.utc) - cycle_dt).total_seconds() / 3600.0
    is_stale = age_hours > stale_threshold_hours
    
    # Try parsing cycle date and hour from filename e.g. gfs_20240715_00z.nc
    cycle_tag = "00Z"
    for part in latest_fname.replace("-", "_").split("_"):
        if len(part) == 3 and part[:2].isdigit() and part[2].lower() == "z":
            cycle_tag = part.upper()
            
    checksum = compute_sha256(latest_fpath) if latest_size < 100 * 1024 * 1024 else "deferred_large_file"
    
    return {
        "provider": provider_clean,
        "status": "HEALTHY" if not is_stale and latest_size > 1024 else ("STALE" if is_stale else "INCOMPLETE"),
        "latest_file": latest_fname,
        "file_path": latest_fpath,
        "latest_cycle": cycle_dt.isoformat(),
        "cycle_hour": cycle_tag,
        "file_size_bytes": latest_size,
        "age_hours": round(age_hours, 2),
        "checksum_sha256": checksum,
        "is_stale": is_stale,
        "available_candidates_count": len(candidates)
    }

def validate_file(filepath: str, provider: str = "GFS") -> Tuple[bool, List[str]]:
    """
    Validates physical integrity of NWP forecast file:
    - File exists and is non-empty (>1 KB)
    - Not corrupted (checks header magic bytes for NetCDF / GRIB2 / CSV)
    - If NetCDF, verifies opening and coordinates
    """
    issues = []
    if not os.path.exists(filepath):
        return False, [f"File not found: {filepath}"]
        
    size = os.path.getsize(filepath)
    if size < 512:
        return False, [f"Corrupted or incomplete file (size {size} bytes is too small)"]
        
    ext = os.path.splitext(filepath)[-1].lower()
    
    # Magic bytes check
    with open(filepath, "rb") as f:
        head = f.read(16)
        
    if ext in [".nc", ".nc4"]:
        # NetCDF3 starts with 'CDF\x01' or 'CDF\x02'; NetCDF4/HDF5 starts with '\x89HDF\r\n\x1a\n'
        is_cdf = head.startswith(b"CDF")
        is_hdf5 = head.startswith(b"\x89HDF")
        if not (is_cdf or is_hdf5):
            issues.append(f"Invalid NetCDF magic bytes in {filepath}")
    elif ext in [".grb2", ".grib2"]:
        if not head.startswith(b"GRIB"):
            issues.append(f"Invalid GRIB2 magic bytes in {filepath}")
            
    # Try xarray check if available
    try:
        import xarray as xr
        if ext in [".nc", ".nc4"]:
            with xr.open_dataset(filepath) as ds:
                if not any(c in ds.coords for c in ["lat", "latitude", "LAT"]):
                    issues.append("Missing latitude coordinate in NetCDF dataset")
                if not any(c in ds.coords for c in ["lon", "longitude", "LON"]):
                    issues.append("Missing longitude coordinate in NetCDF dataset")
    except Exception as e:
        issues.append(f"Corrupted dataset structure: {str(e)}")
        
    return len(issues) == 0, issues

def verify_checksum(filepath: str, expected_sha256: Optional[str] = None) -> Tuple[bool, str]:
    """Verifies file checksum against expected hash."""
    actual_hash = compute_sha256(filepath)
    if expected_sha256:
        is_valid = actual_hash.lower() == expected_sha256.lower()
        return is_valid, actual_hash
    return True, actual_hash

def download_cycle(
    url: str,
    target_path: str,
    expected_sha256: Optional[str] = None,
    max_retries: int = 3,
    timeout_sec: int = 30
) -> Dict[str, Any]:
    """
    Downloads NWP cycle file with retries, timeout, and checksum verification.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = f"{target_path}.tmp_{int(time.time())}"
    
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        attempts += 1
        try:
            logger.info(f"Downloading cycle from {url} (Attempt {attempts}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "MonsoonAI-Ingestion/2.1"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(temp_path, "wb") as out_f:
                while chunk := resp.read(65536):
                    out_f.write(chunk)
                    
            # Check validation
            valid, issues = validate_file(temp_path)
            if not valid:
                raise NWPValidationError(f"Downloaded file validation failed: {'; '.join(issues)}")
                
            if expected_sha256:
                chk_ok, actual_h = verify_checksum(temp_path, expected_sha256)
                if not chk_ok:
                    raise NWPValidationError(f"Checksum mismatch: expected {expected_sha256}, got {actual_h}")
                    
            os.replace(temp_path, target_path)
            return {
                "success": True,
                "file_path": target_path,
                "size_bytes": os.path.getsize(target_path),
                "checksum": compute_sha256(target_path),
                "attempts": attempts
            }
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Download attempt {attempts} failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            time.sleep(1.5 * attempts)
            
    return {
        "success": False,
        "error": f"Failed after {max_retries} attempts: {last_error}",
        "url": url,
        "target_path": target_path
    }

def archive_cycle(
    source_file: str,
    provider: str,
    cycle_dt: datetime.datetime,
    lead_time_hours: int,
    archive_root: str = "results/archive"
) -> str:
    """
    Archives a forecast cycle into a structured, immutable directory tree.
    Does not overwrite existing historical runs.
    """
    year_str = str(cycle_dt.year)
    month_str = f"{cycle_dt.month:02d}"
    day_str = f"{cycle_dt.day:02d}"
    cycle_tag = f"{cycle_dt.hour:02d}Z"
    
    target_dir = os.path.join(archive_root, year_str, month_str, day_str, provider.upper(), cycle_tag, f"{lead_time_hours}h")
    os.makedirs(target_dir, exist_ok=True)
    
    fname = os.path.basename(source_file)
    target_file = os.path.join(target_dir, fname)
    
    if os.path.exists(target_file):
        # Prevent overwrite by appending unique timestamp
        ts = int(time.time())
        target_file = os.path.join(target_dir, f"{os.path.splitext(fname)[0]}_{ts}{os.path.splitext(fname)[1]}")
        
    import shutil
    shutil.copy2(source_file, target_file)
    logger.info(f"Archived forecast cycle to {target_file}")
    return target_file

def return_dataset_manifest(
    nwp_files: List[str],
    obs_files: List[str],
    provider: str,
    cycle: str,
    mode: str = "REAL"
) -> Dict[str, Any]:
    """Generates a complete dataset manifest with cryptographic checksums."""
    nwp_manifest = []
    for f in nwp_files:
        if os.path.exists(f):
            nwp_manifest.append({
                "file": os.path.basename(f),
                "path": f,
                "size_bytes": os.path.getsize(f),
                "sha256": compute_sha256(f)
            })
            
    obs_manifest = []
    for f in obs_files:
        if os.path.exists(f):
            obs_manifest.append({
                "file": os.path.basename(f),
                "path": f,
                "size_bytes": os.path.getsize(f),
                "sha256": compute_sha256(f)
            })
            
    hasher = hashlib.sha256()
    for item in nwp_manifest + obs_manifest:
        hasher.update(item["sha256"].encode("utf-8"))
    combined_hash = hasher.hexdigest()
    
    return {
        "manifest_version": "2.1.0",
        "mode": mode,
        "provider": provider,
        "cycle": cycle,
        "combined_dataset_sha256": combined_hash,
        "nwp_files": nwp_manifest,
        "observation_files": obs_manifest,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

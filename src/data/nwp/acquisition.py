"""
Operational NWP Data Acquisition Pipeline.
Performs remote cycle discovery, HTTP downloading with retry/timeout, checksum generation,
NetCDF validation, and secure disk archival.
"""

import os
import time
import hashlib
import json
import datetime
from typing import Dict, Any, Optional
import requests
import xarray as xr

from .factory import NWPAdapterFactory

class NWPAcquisitionPipeline:
    """
    Automated acquisition and archival orchestrator for operational NWP forecasts.
    """
    def __init__(self, raw_dir: str = "data/raw", timeout_sec: int = 30, max_retries: int = 3):
        self.raw_dir = raw_dir
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        os.makedirs(os.path.join(self.raw_dir, "nwp"), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, "manifests"), exist_ok=True)

    def acquire_forecast_cycle(
        self,
        provider: str = "GFS",
        date_str: Optional[str] = None,
        cycle: str = "00Z",
        lead_time_hours: int = 24,
        force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Executes genuine acquisition sequence for operational NWP data.
        Returns execution manifest with exact temporal provenance and checksums.
        """
        adapter = NWPAdapterFactory.get_adapter(provider, cycle)
        
        if date_str is None:
            latest = adapter.discover_latest_cycle()
            date_str = latest["date"]
            cycle = latest["cycle"]

        date_clean = date_str.replace("-", "")
        dest_filename = f"{provider.lower()}_{date_clean}_{cycle.lower()}_{lead_time_hours}h.nc"
        dest_path = os.path.join(self.raw_dir, "nwp", dest_filename)

        url = adapter.build_download_url(date_str, cycle, lead_time_hours)
        download_needed = force_download or not os.path.exists(dest_path)

        download_status = "CACHED"
        error_msg = None
        sha256_hash = None

        if download_needed:
            # Execute HTTP acquisition with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    res = requests.get(url, timeout=self.timeout_sec, stream=True)
                    if res.status_code == 200:
                        hasher = hashlib.sha256()
                        with open(dest_path, "wb") as f:
                            for chunk in res.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    hasher.update(chunk)
                        sha256_hash = hasher.hexdigest()
                        download_status = "DOWNLOADED"
                        break
                    else:
                        error_msg = f"HTTP {res.status_code} from provider"
                except Exception as e:
                    error_msg = str(e)
                    time.sleep(attempt * 1.5)

            if download_status != "DOWNLOADED":
                download_status = "DOWNLOAD_FAILED"

        # If file exists on disk, compute hash and validate NetCDF
        grid_valid = False
        validation_info = {}
        if os.path.exists(dest_path):
            if not sha256_hash:
                with open(dest_path, "rb") as f:
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
            try:
                ds = xr.open_dataset(dest_path)
                validation_info = adapter.validate_grid(ds)
                grid_valid = validation_info.get("passed", False)
            except Exception as e:
                grid_valid = False
                validation_info = {"passed": False, "error": str(e)}

        init_time = f"{date_str}T{cycle.replace('Z', '')}:00:00Z"
        try:
            valid_dt = datetime.datetime.fromisoformat(init_time.replace("Z", "+00:00")) + datetime.timedelta(hours=lead_time_hours)
            valid_time = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            valid_time = f"{date_str}T{lead_time_hours:02d}:00:00Z"

        manifest = {
            "provider": provider,
            "cycle": cycle,
            "init_time": init_time,
            "valid_time": valid_time,
            "lead_time_hours": lead_time_hours,
            "source_url": url,
            "local_path": dest_path,
            "file_exists": os.path.exists(dest_path),
            "sha256": sha256_hash,
            "status": download_status,
            "error": error_msg,
            "grid_valid": grid_valid,
            "validation": validation_info,
            "acquired_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # Save manifest to disk
        manifest_path = os.path.join(
            self.raw_dir, "manifests", f"manifest_{provider}_{date_clean}_{cycle}_{lead_time_hours}h.json"
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

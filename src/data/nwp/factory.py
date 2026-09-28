"""
Factory for instantiating NWP adapters based on provider name or file metadata.
"""

from typing import Optional
from src.data.nwp.base import BaseNWPAdapter
from src.data.nwp.gfs_adapter import GFSAdapter
from src.data.nwp.ecmwf_adapter import ECMWFAdapter
from src.data.nwp.ncmrwf_adapter import NCMRWFAdapter

def get_nwp_adapter(provider_name: str = "GFS") -> BaseNWPAdapter:
    """
    Factory function to retrieve NWP adapter for a given model.
    Supported: GFS, ECMWF, NCMRWF
    """
    name_clean = provider_name.strip().upper()
    if "ECMWF" in name_clean or "IFS" in name_clean:
        return ECMWFAdapter()
    elif "NCMRWF" in name_clean or "NCUM" in name_clean:
        return NCMRWFAdapter()
    elif "GFS" in name_clean or "NCEP" in name_clean:
        return GFSAdapter()
    else:
        # Default to GFS compatible adapter
        return GFSAdapter()

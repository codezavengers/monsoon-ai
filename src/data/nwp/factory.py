"""
Factory pattern for creating NWP Adapters.
"""

from typing import Dict, Type
from .base import BaseNWPAdapter
from .gfs_adapter import GFSAdapter
from .ecmwf_adapter import ECMWFAdapter
from .ncmrwf_adapter import NCMRWFAdapter

_ADAPTER_REGISTRY: Dict[str, Type[BaseNWPAdapter]] = {
    "GFS": GFSAdapter,
    "GFS_0.25DEG": GFSAdapter,
    "ECMWF": ECMWFAdapter,
    "NCMRWF": NCMRWFAdapter,
}

class NWPAdapterFactory:
    """Instantiates appropriate provider adapter with cycle specifications."""
    @classmethod
    def get_adapter(cls, provider: str, cycle: str = "00Z") -> BaseNWPAdapter:
        p_clean = provider.strip().upper().replace(" 0.25° (NOAA)", "").replace(" 0.25°", "")
        if p_clean in _ADAPTER_REGISTRY:
            return _ADAPTER_REGISTRY[p_clean](cycle=cycle)
        raise ValueError(f"Unknown NWP provider '{provider}'. Supported providers: GFS, ECMWF, NCMRWF")

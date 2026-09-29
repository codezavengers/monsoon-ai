#!/usr/bin/env python3
"""
CLI entry point for SIH PS 26080: Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts.
Supports:
  python3 run.py demo    (Executes benchmark pipeline using synthetic JJAS 2018-2024 dataset)
  python3 run.py real    (Executes operational pipeline using genuine NWP and IMD data)
"""

import sys
import os
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import run_complete_pipeline
from src.utils.logging import get_logger

logger = get_logger("run_cli")

def main():
    parser = argparse.ArgumentParser(
        description="Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "real", "DEMO", "REAL"],
        help="Pipeline execution mode: 'demo' (synthetic benchmark) or 'real' (authentic NWP+IMD)"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    mode = args.mode.upper()
    logger.info(f"Triggering execution: mode={mode}, config={args.config}")
    try:
        metrics = run_complete_pipeline(mode=mode, config_path=args.config)
        logger.info(f"Execution succeeded! Model version: {metrics.get('model_version')}")
        print("\n=== PIPELINE EXECUTION SUMMARY ===")
        print(f"Mode: {metrics.get('mode')}")
        print(f"Domain: {metrics.get('data_provenance', {}).get('domain')}")
        print(f"Generated at: {metrics.get('generated_at')}")
        print(f"Artifacts registered: {len(metrics.get('artifact_hashes', {}))}")
        print("Done.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI and execution entry point for Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts.
Problem Statement ID: 26080.

Commands:
  python run.py demo              Run full end-to-end pipeline and output results
  python run.py generate-data     Generate synthetic meteorological NWP dataset
  python run.py preprocess        Validate and clean raw dataset
  python run.py classify-regimes  Classify weather regimes with rules & ML
  python run.py train             Train baseline, global ML, and regime-aware models
  python run.py predict           Run prediction on sample or test dataset
  python run.py verify            Calculate deterministic, categorical, and FSS verification metrics
  python run.py dashboard         Launch dashboard / API service
"""

import sys
import os
import json
import argparse
from typing import Dict, Any

from src.utils.logging import setup_logger
from src.utils.config import load_config
from src.pipeline import RainfallPostProcessingPipeline
from src.data.loaders import generate_synthetic_monsoon_dataset, save_records_to_csv, load_records_from_csv
from src.data.validation import clean_and_validate_dataset

logger = setup_logger("run_cli")

def cmd_demo(args):
    """Executes full hackathon demo pipeline."""
    print("=" * 80)
    print("DEMO MODE: Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts")
    print("Problem Statement ID: 26080")
    print("=" * 80)
    pipeline = RainfallPostProcessingPipeline()
    results = pipeline.run_full_pipeline()
    
    print("\n--- MODEL COMPARISON TABLE (Heavy Rain Threshold >= 64.5 mm) ---")
    comp = results["model_comparison"]["comparison_table"]
    header = f"{'Model':<30} | {'RMSE':<6} | {'MAE':<6} | {'Bias':<6} | {'CSI':<5} | {'ETS':<5} | {'POD':<5} | {'FAR':<5} | {'FSS':<5}"
    print(header)
    print("-" * len(header))
    for row in comp:
        print(f"{row['model']:<30} | {row['rmse']:<6.2f} | {row['mae']:<6.2f} | {row['bias']:<6.2f} | {row['csi']:<5.3f} | {row['ets']:<5.3f} | {row['pod']:<5.3f} | {row['far']:<5.3f} | {row['fss']:<5.3f}")
        
    print("\n--- REGIME CLASSIFIER ACCURACY ---")
    reg_acc = results["regime_classifier_evaluation"]["accuracy"]
    print(f"Accuracy: {reg_acc * 100:.1f}%")
    
    print("\n--- SAMPLE HIGH-IMPACT DISTRICT FORECASTS ---")
    dists = results["district_forecasts"][:5]
    print(f"{'District':<18} | {'State':<14} | {'Regime':<18} | {'Raw NWP':<8} | {'AI Corr':<8} | {'P(Heavy)':<8} | {'Category'}")
    print("-" * 88)
    for d in dists:
        print(f"{d['district']:<18} | {d['state']:<14} | {d['regime']:<18} | {d['raw_nwp_max']:<8.1f} | {d['corrected_max']:<8.1f} | {d['p_heavy']*100:<7.1f}% | {d['category']}")
        
    print("\nArtifacts saved under 'models/' and 'results/'.")
    print("Run `python run.py dashboard` to view the interactive web interface.")
    return 0

def cmd_generate_data(args):
    """Generates synthetic dataset."""
    logger.info("Generating synthetic meteorological dataset...")
    records = generate_synthetic_monsoon_dataset(start_year=2018, end_year=2024)
    out_path = args.output or "data/synthetic/monsoon_dataset_2018_2024.csv"
    save_records_to_csv(records, out_path)
    logger.info(f"Saved {len(records)} records to {out_path}")
    return 0

def cmd_preprocess(args):
    """Cleans and validates dataset."""
    in_path = args.input or "data/synthetic/monsoon_dataset_2018_2024.csv"
    if not os.path.exists(in_path):
        logger.error(f"Input file not found: {in_path}. Run generate-data first.")
        return 1
    records = load_records_from_csv(in_path)
    cleaned, stats = clean_and_validate_dataset(records)
    out_path = args.output or "data/processed/cleaned_monsoon.csv"
    save_records_to_csv(cleaned, out_path)
    logger.info(f"Preprocessing completed. Stats: {stats}. Saved to {out_path}")
    return 0

def cmd_train(args):
    """Runs training on processed data."""
    pipeline = RainfallPostProcessingPipeline()
    pipeline.run_full_pipeline()
    logger.info("Training finished and models serialized.")
    return 0

def cmd_predict(args):
    """Runs prediction on arbitrary input or sample test file."""
    if not os.path.exists("results/summary_metrics.json"):
        pipeline = RainfallPostProcessingPipeline()
        pipeline.run_full_pipeline()
    with open("results/summary_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Sample prediction for {len(data['district_forecasts'])} Indian districts loaded.")
    print(json.dumps(data["district_forecasts"][:2], indent=2))
    return 0

def cmd_verify(args):
    """Runs verification calculations and prints comparison table."""
    pipeline = RainfallPostProcessingPipeline()
    results = pipeline.run_full_pipeline()
    comp = results["model_comparison"]["comparison_table"]
    print(json.dumps(comp, indent=2))
    return 0

def cmd_dashboard(args):
    """Runs interactive dashboard / web interface."""
    logger.info("Regime-Aware Rainfall AI Dashboard is integrated into the web frontend.")
    logger.info("Access the interactive interface via port 3000 in your browser.")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # demo
    demo_p = subparsers.add_parser("demo", help="Run complete demo pipeline")
    
    # generate-data
    gen_p = subparsers.add_parser("generate-data", help="Generate synthetic NWP & observation data")
    gen_p.add_argument("--output", type=str, default="data/synthetic/monsoon_dataset_2018_2024.csv")
    
    # preprocess
    prep_p = subparsers.add_parser("preprocess", help="Preprocess, validate and clean data")
    prep_p.add_argument("--input", type=str, default="data/synthetic/monsoon_dataset_2018_2024.csv")
    prep_p.add_argument("--output", type=str, default="data/processed/cleaned_monsoon.csv")
    
    # classify-regimes
    subparsers.add_parser("classify-regimes", help="Classify weather regimes")
    
    # train
    subparsers.add_parser("train", help="Train baseline and regime-aware ML models")
    
    # predict
    subparsers.add_parser("predict", help="Generate predictions")
    
    # verify
    subparsers.add_parser("verify", help="Calculate meteorological verification metrics")
    
    # dashboard
    subparsers.add_parser("dashboard", help="Start web dashboard")
    
    args = parser.parse_args()
    
    if args.command == "demo" or args.command is None:
        return cmd_demo(args)
    elif args.command == "generate-data":
        return cmd_generate_data(args)
    elif args.command == "preprocess":
        return cmd_preprocess(args)
    elif args.command in ["train", "classify-regimes"]:
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "dashboard":
        return cmd_dashboard(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())

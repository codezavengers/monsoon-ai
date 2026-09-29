"""
Dataset loaders and temporal train/val/test splitting for monsoon data.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

def load_monsoon_dataset(csv_path: str = "data/synthetic/monsoon_dataset_2018_2024.csv") -> pd.DataFrame:
    """Loads the JJAS monsoon benchmark dataset (2018-2024)."""
    if not os.path.exists(csv_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        alt = os.path.join(base_dir, csv_path)
        if os.path.exists(alt):
            csv_path = alt

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Monsoon dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df

def load_train_val_test_splits(
    csv_path: str = "data/synthetic/monsoon_dataset_2018_2024.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns temporal splits according to the project specification:
    - Train: 2018-2022 (JJAS)
    - Validation: 2023 (JJAS)
    - Test: 2024 (JJAS)
    """
    df = load_monsoon_dataset(csv_path)
    train_df = df[df["year"].isin([2018, 2019, 2020, 2021, 2022])].copy()
    val_df = df[df["year"] == 2023].copy()
    test_df = df[df["year"] == 2024].copy()
    return train_df, val_df, test_df

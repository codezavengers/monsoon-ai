"""
Reproducibility utilities for random seeds.
"""

import os
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across standard Python and NumPy for reproducible results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

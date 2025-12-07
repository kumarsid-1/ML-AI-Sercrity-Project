import os
import random
import sys

import numpy as np
import torch

from src.logger import logging
from src.exception import CustomException

# -------------------------------------------------------------------
# GLOBAL SETTINGS
# -------------------------------------------------------------------

SEED: int = 42
DEVICE = torch.device("cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "results")


def init_directories():
    """
    Create results directory and log any filesystem issues.
    """
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        logging.info(f"Output directory initialized at: {OUT_DIR}")
    except Exception as e:
        raise CustomException(e, sys)


def set_seed(seed: int = SEED) -> None:
    """
    Set global random seed for reproducibility.
    """
    try:
        global SEED
        SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f"Random seed set to: {seed}")
    except Exception as e:
        raise CustomException(e, sys)


# Initialize on import
try:
    init_directories()
except Exception as e:
    # This will already be logged in CustomException
    raise

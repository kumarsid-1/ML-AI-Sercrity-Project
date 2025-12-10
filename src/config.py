import os
import random
import sys
import numpy as np
import torch
from src.logger import logging
from src.exception import CustomException


# GLOBAL SETTINGS
SEED: int = 50
DEVICE = torch.device("cpu")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "results")


logging.info(f"Device used: {DEVICE}")
logging.info(f"Project root: {PROJECT_ROOT}")



# Setting Global seed
def set_seed(seed: int = SEED) -> None:
    try:
        global SEED
        SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f"Random seed set to: {seed}")
    except Exception as e:
        raise CustomException(e, sys)


# Creating results directory if it does not exists and log any system errors
def init_directories():
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        logging.info(f"Output directory initialized at: {OUT_DIR}")
    except Exception as e:
        raise CustomException(e, sys)

        
# Initialize on import
try:
    init_directories()
except Exception as e:
    raise

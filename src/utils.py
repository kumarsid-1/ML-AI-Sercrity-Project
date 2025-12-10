import os
import sys
import numpy as np
from typing import Any
from matplotlib.figure import Figure
from src.config import OUT_DIR
from src.logger import logging
from src.exception import CustomException


# Computing Population Stability Index (PSI) between expected and actual distributions.
def compute_psi(expected: Any, actual: Any, bins: int = 10) -> float:
    logging.info("Computing PSI")
    try:
        expected = np.array(expected)
        actual = np.array(actual)
        breaks = np.linspace(expected.min(), expected.max(), bins + 1)
        e_ct, _ = np.histogram(expected, bins=breaks)
        a_ct, _ = np.histogram(actual, bins=breaks)
        e_pct = e_ct / (len(expected) + 1e-8)
        a_pct = a_ct / (len(actual) + 1e-8)
        e_pct = np.where(e_pct == 0, 1e-6, e_pct)
        a_pct = np.where(a_pct == 0, 1e-6, a_pct)
        psi = np.sum((e_pct - a_pct) * np.log(e_pct / a_pct))
        logging.info(f"PSI is: {psi:.6f}")
        return float(psi)
    except Exception as e:
        raise CustomException(e, sys)


# Compatibility wrapper for different river ADWIN versions to check if any change is detected. 
def adwin_change_detected(adwin) -> bool:
    try:
        if hasattr(adwin, "change_detected"):
            atr = adwin.change_detected
            return atr() if callable(atr) else bool(atr)
        if hasattr(adwin, "detected_change"):
            atr = adwin.detected_change
            return atr() if callable(atr) else bool(atr)
        return False
    except Exception as e:
        raise CustomException(e, sys)


# Save figures to the result directory and return path
def save_fig(fig: Figure, name: str) -> str:
    try:
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"The Figure is: {path}")
        return path
    except Exception as e:
        raise CustomException(e, sys)
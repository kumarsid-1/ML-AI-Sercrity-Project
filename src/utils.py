import os
import sys
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.config import OUT_DIR
from src.logger import logging
from src.exception import CustomException


def save_fig(fig: Figure, name: str) -> str:
    """
    Save figure to results directory and return path.
    """
    try:
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"Figure saved: {path}")
        return path
    except Exception as e:
        raise CustomException(e, sys)


def compute_psi(expected: Any, actual: Any, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    """
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
        logging.info(f"Computed PSI: {psi:.6f}")
        return float(psi)
    except Exception as e:
        raise CustomException(e, sys)


def adwin_change_detected(adwin) -> bool:
    """
    Compatibility wrapper for different river ADWIN versions.
    """
    try:
        if hasattr(adwin, "change_detected"):
            attr = adwin.change_detected
            return attr() if callable(attr) else bool(attr)
        if hasattr(adwin, "detected_change"):
            attr = adwin.detected_change
            return attr() if callable(attr) else bool(attr)
        return False
    except Exception as e:
        raise CustomException(e, sys)

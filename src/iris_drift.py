import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

from river.drift import ADWIN

from src.logger import logging
from src.exception import CustomException

from src.config import (
    DRIFT_CONFIG,
    RESULTS_DIR
)

# =============================================================================
# POPULATION STABILITY INDEX (PSI)
# =============================================================================

def population_stability_index(
    expected,
    actual,
    bins=10
):

    """
    Computes Population Stability Index (PSI)
    for statistical drift measurement.
    """

    try:

        expected_percents, bin_edges = np.histogram(
            expected,
            bins=bins
        )

        actual_percents, _ = np.histogram(
            actual,
            bins=bin_edges
        )

        expected_percents = (
            expected_percents / len(expected)
        )

        actual_percents = (
            actual_percents / len(actual)
        )

        psi = np.sum(
            (
                actual_percents - expected_percents
            )
            * np.log(
                (
                    actual_percents + 1e-8
                )
                / (
                    expected_percents + 1e-8
                )
            )
        )

        return float(psi)

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# INDUCE CONTROLLED DRIFT
# =============================================================================

def induce_drift(
    dataframe
):

    """
    Injects controlled synthetic drift
    into Iris dataset.

    Research Motivation:
    --------------------
    Controlled drift allows:
    - reproducible experiments
    - sensitivity analysis
    - statistical validation
    """

    try:

        logging.info(
            "Injecting controlled statistical drift"
        )

        feature_name = DRIFT_CONFIG["feature_name"]

        shift_constant = DRIFT_CONFIG["shift_constant"]

        drift_portion = DRIFT_CONFIG["drift_portion"]

        df = dataframe.copy()

        split_index = int(
            len(df) * (1 - drift_portion)
        )

        df.loc[
            split_index:,
            feature_name
        ] += shift_constant

        logging.info(
            f"Drift injected | "
            f"Feature: {feature_name} | "
            f"Shift: {shift_constant} | "
            f"Portion affected: {drift_portion}"
        )

        return df

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# KS TEST
# =============================================================================

def run_ks_test(
    reference,
    shifted
):

    """
    Performs Kolmogorov-Smirnov test
    for drift detection.
    """

    try:

        ks_statistic, p_value = ks_2samp(
            reference,
            shifted
        )

        return {
            "ks_statistic": float(ks_statistic),
            "ks_p_value": float(p_value)
        }

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# ADWIN STREAMING DRIFT DETECTION
# =============================================================================

def run_adwin_detection(
    stream_data
):

    """
    Performs streaming drift detection
    using ADWIN.
    """

    try:

        detector = ADWIN()

        drift_points = []

        for idx, value in enumerate(stream_data):

            detector.update(value)

            if detector.drift_detected:

                drift_points.append(idx)

        return {
            "drift_detected": len(drift_points) > 0,
            "drift_points": drift_points
        }

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# VISUALIZE DRIFT
# =============================================================================

def save_drift_visualization(
    original,
    shifted
):

    """
    Creates publication-quality
    drift visualization.
    """

    try:

        plt.figure(figsize=(8, 5))

        plt.hist(
            original,
            bins=20,
            alpha=0.6,
            label="Original"
        )

        plt.hist(
            shifted,
            bins=20,
            alpha=0.6,
            label="Shifted"
        )

        plt.xlabel("Feature Values")

        plt.ylabel("Frequency")

        plt.title(
            "Distribution Shift in Iris Dataset"
        )

        plt.legend()

        save_path = os.path.join(
            RESULTS_DIR,
            "iris_drift_visualization.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logging.info(
            f"Saved drift visualization: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# COMPLETE DRIFT PIPELINE
# =============================================================================

def run_drift_pipeline():

    """
    Full drift analysis pipeline:
    - load dataset
    - inject drift
    - compute KS-Test
    - compute PSI
    - run ADWIN
    - generate visualizations
    - save structured report
    """

    try:

        logging.info(
            "Starting drift detection pipeline"
        )

        start_time = time.time()

        # =============================================================
        # LOAD DATASET
        # =============================================================

        iris = load_iris(as_frame=True)

        dataframe = iris.frame

        feature_name = DRIFT_CONFIG["feature_name"]

        original_values = dataframe[
            feature_name
        ].values

        # =============================================================
        # DRIFT INJECTION
        # =============================================================

        shifted_dataframe = induce_drift(
            dataframe
        )

        shifted_values = shifted_dataframe[
            feature_name
        ].values

        # =============================================================
        # KS TEST
        # =============================================================

        ks_results = run_ks_test(
            original_values,
            shifted_values
        )

        # =============================================================
        # PSI
        # =============================================================

        psi_score = population_stability_index(
            original_values,
            shifted_values
        )

        # =============================================================
        # ADWIN
        # =============================================================

        adwin_results = run_adwin_detection(
            shifted_values
        )

        # =============================================================
        # VISUALIZATION
        # =============================================================

        save_drift_visualization(
            original_values,
            shifted_values
        )

        runtime = time.time() - start_time

        # =============================================================
        # STRUCTURED REPORT
        # =============================================================

        results = {
            "drift_configuration": DRIFT_CONFIG,
            "ks_test": ks_results,
            "psi_score": psi_score,
            "adwin_results": adwin_results,
            "runtime_seconds": runtime
        }

        save_path = os.path.join(
            RESULTS_DIR,
            "iris_drift_report.json"
        )

        with open(save_path, "w") as file:

            json.dump(
                results,
                file,
                indent=4
            )

        logging.info(
            f"Saved drift report: {save_path}"
        )

        logging.info(
            "Drift detection pipeline completed"
        )

        return results

    except Exception as e:

        raise CustomException(e, sys)
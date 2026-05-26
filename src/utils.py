import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from src.logger import logging
from src.exception import CustomException

from src.config import RESULTS_DIR

# =============================================================================
# CUSTOM JSON ENCODER
# =============================================================================

class NumpyEncoder(json.JSONEncoder):

    """
    Converts:
    - NumPy objects
    - Torch tensors
    - Torch devices

    into JSON serializable formats.
    """

    def default(
        self,
        obj
    ):

        import torch
        import numpy as np

        # =============================================================
        # NUMPY TYPES
        # =============================================================

        if isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.ndarray):

            return obj.tolist()

        # =============================================================
        # TORCH TENSORS
        # =============================================================

        elif isinstance(obj, torch.Tensor):

            return obj.detach().cpu().tolist()

        # =============================================================
        # TORCH DEVICE
        # =============================================================

        elif isinstance(obj, torch.device):

            return str(obj)

        return super().default(obj)

# =============================================================================
# SAVE JSON FILE
# =============================================================================

def save_json(
    data,
    save_path
):

    """
    Saves structured JSON reports.

    Handles:
    - tensors
    - numpy arrays
    - torch devices
    """

    try:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        with open(save_path, "w") as file:

            json.dump(
                data,
                file,
                indent=4,
                cls=NumpyEncoder
            )

        logging.info(
            f"JSON file saved successfully: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# LOAD JSON FILE
# =============================================================================

def load_json(
    file_path
):

    """
    Loads JSON report files.
    """

    try:

        with open(file_path, "r") as file:

            data = json.load(file)

        logging.info(
            f"JSON file loaded successfully: {file_path}"
        )

        return data

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SAVE HIGH-RES FIGURE
# =============================================================================

def save_figure(
    figure,
    filename
):

    """
    Saves publication-quality figures.
    """

    try:

        save_path = os.path.join(
            RESULTS_DIR,
            filename
        )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        logging.info(
            f"Figure saved successfully: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# CREATE ATTACK COMPARISON PLOT
# =============================================================================

def create_attack_comparison_plot(
    results_dict,
    save_name="attack_comparison.png"
):

    """
    Generates attack comparison chart.
    """

    try:

        attacks = list(results_dict.keys())

        accuracies = [
            results_dict[attack]["accuracy"]
            for attack in attacks
        ]

        fig = plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            attacks,
            accuracies
        )

        plt.xlabel(
            "Attack Type"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            "Model Accuracy Under Adversarial Attacks"
        )

        save_figure(
            fig,
            save_name
        )

        plt.close()

        logging.info(
            "Attack comparison plot created"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# CREATE DRIFT DISTRIBUTION PLOT
# =============================================================================

def create_drift_distribution_plot(
    original,
    shifted,
    save_name="drift_distribution.png"
):

    """
    Visualizes original vs shifted distributions.
    """

    try:

        fig = plt.figure(
            figsize=(8, 5)
        )

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

        plt.xlabel(
            "Feature Values"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            "Statistical Drift Visualization"
        )

        plt.legend()

        save_figure(
            fig,
            save_name
        )

        plt.close()

        logging.info(
            "Drift visualization created"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SAVE EXPERIMENT METADATA
# =============================================================================

def save_experiment_metadata(
    metadata,
    filename="experiment_metadata.json"
):

    """
    Saves experiment tracking metadata.
    """

    try:

        save_path = os.path.join(
            RESULTS_DIR,
            filename
        )

        save_json(
            metadata,
            save_path
        )

        logging.info(
            "Experiment metadata saved"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# INITIALIZE RESULTS DIRECTORY
# =============================================================================

def initialize_results_structure():

    """
    Creates standardized result directories.
    """

    try:

        directories = [

            "results",
            "results/plots",
            "results/reports",
            "results/adversarial",
            "results/drift"
        ]

        for directory in directories:

            os.makedirs(
                directory,
                exist_ok=True
            )

        logging.info(
            "Results directory structure initialized"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# PRINT EXPERIMENT SUMMARY
# =============================================================================

def print_experiment_summary(
    summary
):

    """
    Displays structured pipeline summary.
    """

    try:

        print("\n" + "=" * 70)

        print("AI SECURITY PIPELINE SUMMARY")

        print("=" * 70)

        for key, value in summary.items():

            print(f"\n{key}:\n{value}")

        print("\n" + "=" * 70)

        logging.info(
            "Experiment summary displayed"
        )

    except Exception as e:

        raise CustomException(e, sys)
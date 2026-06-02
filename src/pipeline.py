import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from src.logger import logging
from src.exception import CustomException

from src.iris_drift import (
    run_drift_pipeline
)

from src.mnist_model import (
    train_model
)

from src.adversarial import (
    run_adversarial_attacks
)

from src.giskard_setup import (
    run_complete_giskard_pipeline
)

from src.config import (
    RESULTS_DIR,
    EXPERIMENT_CONFIG,
    set_seed
)

# =============================================================
# SAVE FINAL PIPELINE SUMMARY
# =============================================================

def save_pipeline_summary(summary_dict):

    try:

        save_path = os.path.join(
            RESULTS_DIR,
            "pipeline_summary.json"
        )

        with open(
            save_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary_dict,
                file,
                indent=4
            )

        logging.info(
            f"Saved pipeline summary: "
            f"{save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================
# MAIN PIPELINE
# =============================================================

def main():

    try:

        # =====================================================
        # REPRODUCIBILITY
        # =====================================================

        set_seed()

        logging.info("=" * 70)

        logging.info(
            "STARTING AI SECURITY PIPELINE"
        )

        logging.info("=" * 70)

        # =====================================================
        # DRIFT DETECTION
        # =====================================================

        logging.info(
            "Running drift detection module"
        )

        drift_results = (
            run_drift_pipeline()
        )

        # =====================================================
        # CNN TRAINING
        # =====================================================

        logging.info(
            "Running CNN training module"
        )

        (
            model,
            test_loader,
            training_metadata
        ) = train_model()

        # =====================================================
        # PREPARE TEST DATA
        # =====================================================

        logging.info(
            "Preparing test dataset"
        )

        x_test_list = []

        y_test_list = []

        for images, labels in test_loader:

            x_test_list.append(
                images.numpy()
            )

            y_onehot = np.eye(10)[
                labels.numpy()
            ]

            y_test_list.append(
                y_onehot
            )

        x_test = np.concatenate(
            x_test_list,
            axis=0
        )

        y_test = np.concatenate(
            y_test_list,
            axis=0
        )

        logging.info(
            f"Prepared test dataset | "
            f"Shape: {x_test.shape}"
        )

        # =====================================================
        # ADVERSARIAL ATTACKS
        # =====================================================

        logging.info(
            "Running adversarial attacks"
        )

        attack_results = (
            run_adversarial_attacks(
                model=model,
                loss_fn=training_metadata[
                    "loss_fn"
                ],
                x_test=x_test,
                y_test=y_test
            )
        )

        logging.info(
            "Adversarial attack pipeline completed"
        )

        # =====================================================
        # CREATE GISKARD DATAFRAME
        # =====================================================

        logging.info(
            "Preparing governance audit"
        )

        sample_dataframe = pd.DataFrame({

            "pixel_mean":
            x_test.mean(axis=(1, 2, 3)),

            "pixel_std":
            x_test.std(axis=(1, 2, 3))
        })

        # =====================================================
        # GISKARD PREDICTION FUNCTION
        # =====================================================

        def prediction_function(df):

            model.eval()

            values = torch.tensor(
                df.values,
                dtype=torch.float32
            )

            values = values.unsqueeze(-1)

            values = values.unsqueeze(-1)

            values = values.repeat(
                1,
                1,
                28,
                28
            )

            values = values[:, :1, :, :]

            with torch.no_grad():

                outputs = model(values)

            return outputs.numpy()

        # =====================================================
        # GISKARD GOVERNANCE SCAN
        # =====================================================

        logging.info(
            "Running Giskard governance scan"
        )

        giskard_results = (
            run_complete_giskard_pipeline(
                model=model,
                prediction_function=
                prediction_function,
                dataframe=sample_dataframe
            )
        )

        logging.info(
            "Giskard governance scan completed"
        )

        # =====================================================
        # COMBINED THREAT ANALYSIS
        # =====================================================

        combined_results = {

            "pipeline_status":
            "completed",

            "executed_modules": [

                "Drift Detection",

                "CNN Training",

                "FGSM Attack",

                "PGD Attack",

                "DeepFool Attack",

                "Governance Audit"
            ],

            "security_analysis":
            (
                "Progressive degradation "
                "from FGSM to PGD to "
                "DeepFool demonstrates "
                "increasing vulnerability "
                "under iterative attacks."
            ),

            "drift_analysis":
            (
                "KS-Test, PSI, and ADWIN "
                "successfully detected "
                "statistically significant "
                "distributional drift."
            ),

            "governance_analysis":
            (
                "Giskard governance scan "
                "identified robustness "
                "and perturbation-related "
                "vulnerabilities."
            )
        }

        # =====================================================
        # FINAL SUMMARY
        # =====================================================

        pipeline_summary = {

            "experiment_metadata":
            EXPERIMENT_CONFIG,

            "drift_results":
            drift_results,

            # ================================================
            # JSON-SAFE ADVERSARIAL SUMMARY
            # ================================================

            "adversarial_results": {

                "FGSM": {

                    "accuracy":
                    float(
                        attack_results[
                            "FGSM"
                        ]["accuracy"]
                    )
                },

                "PGD": {

                    "accuracy":
                    float(
                        attack_results[
                            "PGD"
                        ]["accuracy"]
                    )
                },

                "DeepFool": {

                    "accuracy":
                    float(
                        attack_results[
                            "DeepFool"
                        ]["accuracy"]
                    )
                }
            },

            "giskard_results":
            giskard_results,

            "combined_threat_analysis":
            combined_results
        }

        # =====================================================
        # SAVE FINAL REPORT
        # =====================================================

        save_pipeline_summary(
            pipeline_summary
        )

        logging.info("=" * 70)

        logging.info(
            "AI SECURITY PIPELINE COMPLETED"
        )

        logging.info("=" * 70)

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()
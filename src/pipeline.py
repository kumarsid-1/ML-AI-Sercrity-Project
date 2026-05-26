import os
import sys
import json
from datetime import datetime

from src.logger import logging
from src.exception import CustomException

from src.config import (
    SEED,
    DEVICE,
    RESULTS_DIR,
    EXPERIMENT_CONFIG,
    FGSM_EPSILON,
    PGD_CONFIG,
    DEEPFOOL_CONFIG,
    set_seed
)

from src.mnist_model import (
    train_model,
    evaluate_model
)

from src.iris_drift import (
    run_drift_pipeline
)

from src.adversarial import (
    run_fgsm_attack,
    run_pgd_attack,
    run_deepfool_attack,
    combined_threat_evaluation,
    save_attack_comparison_plot,
    save_attack_report
)

from src.utils import save_json

# =============================================================================
# PIPELINE START
# =============================================================================

def main():

    """
    Complete AI Security Pipeline

    Modules:
    --------
    1. Drift Detection
    2. CNN Training
    3. Clean Accuracy Evaluation
    4. FGSM Evaluation
    5. PGD Evaluation
    6. DeepFool Evaluation
    7. Combined Threat Evaluation
    8. Structured Reporting
    9. Visualization Generation
    """

    try:

        # =============================================================
        # REPRODUCIBILITY
        # =============================================================

        set_seed(SEED)

        logging.info(
            "=" * 70
        )

        logging.info(
            "STARTING AI SECURITY PIPELINE"
        )

        logging.info(
            "=" * 70
        )

        # =============================================================
        # DRIFT DETECTION PIPELINE
        # =============================================================

        logging.info(
            "Running drift detection module"
        )

        drift_results = run_drift_pipeline()

        # =============================================================
        # MODEL TRAINING
        # =============================================================

        logging.info(
            "Running CNN training module"
        )

        (
            model,
            test_loader,
            training_metadata
        ) = train_model()

        # =============================================================
        # CLEAN EVALUATION
        # =============================================================

        logging.info(
            "Evaluating clean model performance"
        )

        clean_accuracy = evaluate_model(
            model,
            test_loader
        )

        # =============================================================
        # FGSM ATTACK
        # =============================================================

        logging.info(
            "Running FGSM attack evaluation"
        )

        fgsm_results = run_fgsm_attack(
            model,
            test_loader
        )

        # =============================================================
        # PGD ATTACK
        # =============================================================

        logging.info(
            "Running PGD attack evaluation"
        )

        pgd_results = run_pgd_attack(
            model,
            test_loader
        )

        # =============================================================
        # DEEPFOOL ATTACK
        # =============================================================

        logging.info(
            "Running DeepFool attack evaluation"
        )

        deepfool_results = run_deepfool_attack(
            model,
            test_loader
        )

        # =============================================================
        # COMBINED THREAT EVALUATION
        # =============================================================

        logging.info(
            "Running combined threat evaluation"
        )

        combined_results = (
            combined_threat_evaluation(
                model,
                test_loader
            )
        )

        # =============================================================
        # ATTACK SUMMARY
        # =============================================================

        attack_results = {
            "FGSM": fgsm_results,
            "PGD": pgd_results,
            "DeepFool": deepfool_results
        }

        # =============================================================
        # SAVE ATTACK VISUALIZATION
        # =============================================================

        save_attack_comparison_plot(
            attack_results
        )

        # =============================================================
        # SAVE ATTACK REPORT
        # =============================================================

        save_attack_report(
            attack_results
        )

        # =============================================================
        # FINAL PIPELINE SUMMARY
        # =============================================================

        pipeline_summary = {

            # =========================================================
            # EXPERIMENT METADATA
            # =========================================================

            "experiment_metadata": {

                "timestamp": datetime.now().isoformat(),

                "project_name": (
                    EXPERIMENT_CONFIG["project_name"]
                ),

                "version": (
                    EXPERIMENT_CONFIG["version"]
                ),

                "device": str(DEVICE),

                "seed": SEED,

                "research_focus": (
                    EXPERIMENT_CONFIG["research_focus"]
                )
            },

            # =========================================================
            # TRAINING CONFIGURATION
            # =========================================================

            "training_metadata": training_metadata,

            # =========================================================
            # DRIFT RESULTS
            # =========================================================

            "drift_detection_results": drift_results,

            # =========================================================
            # CLEAN ACCURACY
            # =========================================================

            "clean_model_accuracy": clean_accuracy,

            # =========================================================
            # ADVERSARIAL RESULTS
            # =========================================================

            "adversarial_results": {

                "fgsm": fgsm_results,

                "pgd": pgd_results,

                "deepfool": deepfool_results
            },

            # =========================================================
            # ATTACK CONFIGURATIONS
            # =========================================================

            "attack_configurations": {

                "fgsm": {
                    "epsilon": FGSM_EPSILON
                },

                "pgd": PGD_CONFIG,

                "deepfool": DEEPFOOL_CONFIG
            },

            # =========================================================
            # COMBINED THREAT ANALYSIS
            # =========================================================

            "combined_threat_evaluation": (
                combined_results
            )
        }

        # =============================================================
        # SAVE FINAL REPORT
        # =============================================================

        summary_save_path = os.path.join(
            RESULTS_DIR,
            "pipeline_summary.json"
        )

        save_json(
            pipeline_summary,
            summary_save_path
        )

        logging.info(
            f"Pipeline summary saved: "
            f"{summary_save_path}"
        )

        # =============================================================
        # PIPELINE COMPLETED
        # =============================================================

        logging.info(
            "=" * 70
        )

        logging.info(
            "AI SECURITY PIPELINE COMPLETED SUCCESSFULLY"
        )

        logging.info(
            "=" * 70
        )

        return pipeline_summary

    except Exception as e:

        raise CustomException(
            e,
            sys
        )

# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
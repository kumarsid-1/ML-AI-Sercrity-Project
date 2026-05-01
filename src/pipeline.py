import json
import os
import sys
from src.config import DEVICE, OUT_DIR, SEED, set_seed
from src.exception import CustomException
from src.logger import logging
from src.iris_drift import iris_pipeline
from src.mnist_model import train_mnist
from src.adversarial import comprehensive_adversarial_eval


def main():
    """
    Main pipeline orchestrating all experiments:
    1. Iris drift detection
    2. MNIST training
    3. Comprehensive adversarial evaluation (FGSM, PGD, DeepFool)
    """
    logger = logging.getLogger("pipeline_main")
    try:
        set_seed(SEED)
        logger.info("="*70)
        logger.info("ML-AI-SECURITY-PROJECT: PIPELINE STARTING")
        logger.info("="*70)
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Random seed: {SEED}")
        logger.info(f"Output directory: {OUT_DIR}")

        # ==================== IRIS DRIFT DETECTION ====================
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: IRIS DRIFT DETECTION")
        logger.info("="*70)
        iris_summary = iris_pipeline()
        logger.info("Iris drift detection completed successfully")

        # ==================== MNIST TRAINING ====================
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: MNIST CNN TRAINING")
        logger.info("="*70)
        model, test_loader, mnist_acc = train_mnist(epochs=3)
        logger.info(f"MNIST training completed | Test accuracy: {mnist_acc:.5f}")

        # ==================== ADVERSARIAL ROBUSTNESS EVALUATION ====================
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: ADVERSARIAL ROBUSTNESS EVALUATION")
        logger.info("="*70)
        adv_results = comprehensive_adversarial_eval(model, test_loader)
        logger.info("Comprehensive adversarial evaluation completed")

        # ==================== FINAL SUMMARY ====================
        summary = {
            "project": "ML-AI-Security-Project",
            "authors": ["Siddharth Kumar", "Siddhanth Harish Bist"],
            "iris_drift_detection": {
                "baseline_accuracy": round(iris_summary["baseline"], 5),
                "ks_statistic": round(iris_summary["ks_stat"], 5),
                "ks_p_value": round(iris_summary["ks_p"], 6),
                "psi": round(iris_summary["psi"], 5),
                "adwin_first_detection": iris_summary["adwin_first"],
                "adwin_total_detections": iris_summary["adwin_total"],
            },
            "mnist_classification": {
                "test_accuracy": round(mnist_acc, 5),
            },
            "adversarial_robustness": {
                "clean_accuracy": round(adv_results["fgsm"]["clean"], 5),
                "fgsm_attack": {
                    "adversarial_accuracy": round(adv_results["fgsm"]["adversarial"], 5),
                    "accuracy_drop": round(adv_results["fgsm"]["clean"] - adv_results["fgsm"]["adversarial"], 5),
                },
                "pgd_attack": {
                    "adversarial_accuracy": round(adv_results["pgd"]["adversarial"], 5),
                    "accuracy_drop": round(adv_results["pgd"]["clean"] - adv_results["pgd"]["adversarial"], 5),
                },
                "deepfool_attack": {
                    "adversarial_accuracy": round(adv_results["deepfool"]["adversarial"], 5),
                    "accuracy_drop": round(adv_results["deepfool"]["clean"] - adv_results["deepfool"]["adversarial"], 5),
                },
                "strongest_attack": min(
                    [
                        ("FGSM", adv_results["fgsm"]["adversarial"]),
                        ("PGD", adv_results["pgd"]["adversarial"]),
                        ("DeepFool", adv_results["deepfool"]["adversarial"])
                    ],
                    key=lambda x: x[1]
                )[0]
            }
        }

        summary_path = os.path.join(OUT_DIR, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Final summary saved to: {summary_path}")
        logger.info(f"All results available in: {OUT_DIR}")

    except CustomException:
        raise
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
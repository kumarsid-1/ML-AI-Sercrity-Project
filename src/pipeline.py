import json
import os
import sys

from src.config import DEVICE, OUT_DIR, SEED, set_seed
from src.exception import CustomException
from src.logger import logging
from src.iris_drift import iris_pipeline
from src.mnist_model import train_mnist
from src.adversarial import adversarial_eval


def main():
    logger = logging.getLogger("pipeline_main")
    try:
        set_seed(SEED)

        logger.info("=" * 60)
        logger.info("ML SAFETY PROJECT - COMPLETE PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Random seed: {SEED}")
        logger.info(f"Output directory: {OUT_DIR}")

        iris_summary = iris_pipeline()
        logger.info("Completed Iris drift pipeline")

        model, test_loader, mnist_acc = train_mnist(epochs=3)
        logger.info("Completed MNIST training")

        clean_acc, adv_acc = adversarial_eval(model, test_loader, eps=0.2)
        logger.info("Completed adversarial evaluation")

        summary = {
            "iris": {
                "baseline_accuracy": iris_summary["baseline"],
                "ks_statistic": iris_summary["ks_stat"],
                "ks_p_value": iris_summary["ks_p"],
                "psi": iris_summary["psi"],
                "adwin_first_detection": iris_summary["adwin_first"],
                "adwin_total_detections": iris_summary["adwin_total"],
            },
            "mnist": {
                "test_accuracy": mnist_acc,
                "clean_accuracy": clean_acc,
                "adversarial_accuracy": adv_acc,
                "accuracy_drop": clean_acc - adv_acc,
            },
        }

        summary_path = os.path.join(OUT_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Summary saved to: {summary_path}")

    except CustomException:
        # Already logged in CustomException; re-raise to stop execution
        raise
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

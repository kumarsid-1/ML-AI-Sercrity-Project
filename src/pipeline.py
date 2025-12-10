import json
import os
import sys
from src.config import DEVICE, OUT_DIR, SEED, set_seed
from src.exception import CustomException
from src.logger import logging
from src.iris_drift import iris_pipeline
from src.mnist_model import train_mnist
from src.adversarial import adversarial_eval

# Flow Control
def main():
    logger = logging.getLogger("pipeline_main")
    try:
        set_seed(SEED)
        logger.info("PROJECT STARTING")
        logger.info(f"Device used: {DEVICE}")
        logger.info(f"Random seed set to: {SEED}")
        logger.info(f"Output directory is: {OUT_DIR}")


        logger.info("Starting Iris drift")
        iris_summary = iris_pipeline()
        logger.info("Iris drift Completed")


        logger.info("Starting MNIST training")
        model, test_loader, mnist_acc = train_mnist(epochs=3)
        logger.info("MNIST training Completed")


        logger.info("Starting adversarial evaluation")
        clean_acc, adv_acc = adversarial_eval(model, test_loader, eps=0.2)
        logger.info("Adversarial evaluation completed")

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

        
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Summary saved to: {summary_path}")


    except CustomException:
        raise
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

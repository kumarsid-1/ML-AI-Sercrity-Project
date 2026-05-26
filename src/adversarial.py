import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DeepFool

from art.estimators.classification import PyTorchClassifier

from src.logger import logging
from src.exception import CustomException

from src.config import (
    DEVICE,
    FGSM_EPSILON,
    PGD_CONFIG,
    DEEPFOOL_CONFIG,
    RESULTS_DIR
)

# =============================================================================
# CREATE ART CLASSIFIER
# =============================================================================

def get_art_classifier(model):

    """
    Wraps the PyTorch model using ART classifier
    for adversarial robustness evaluation.
    """

    try:

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
            clip_values=(0, 1),
            device_type=DEVICE
        )

        return classifier

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SINGLE ATTACK EVALUATION
# =============================================================================

def evaluate_single_attack(
    model,
    classifier,
    test_loader,
    attack_name,
    attack_object
):

    """
    Evaluates model robustness under a single adversarial attack.
    """

    try:

        logging.info(f"Starting {attack_name} attack evaluation")

        start_time = time.time()

        correct = 0
        total = 0

        adv_examples = []

        model.eval()

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            np_images = images.cpu().numpy()

            adv_images = attack_object.generate(
                x=np_images
            )

            adv_tensor = torch.tensor(
                adv_images,
                dtype=torch.float32
            ).to(DEVICE)

            outputs = model(adv_tensor)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            if len(adv_examples) < 5:

                adv_examples.append(
                    (
                        images[0].cpu(),
                        adv_tensor[0].cpu(),
                        labels[0].item(),
                        preds[0].item()
                    )
                )

        accuracy = correct / total

        runtime = time.time() - start_time

        logging.info(
            f"{attack_name} evaluation completed | "
            f"Accuracy: {accuracy:.4f}"
        )

        return {
            "attack_name": attack_name,
            "accuracy": accuracy,
            "runtime_seconds": runtime,
            "examples": adv_examples
        }

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# FGSM ATTACK
# =============================================================================

def run_fgsm_attack(
    model,
    test_loader
):

    """
    Runs FGSM adversarial evaluation.
    """

    try:

        classifier = get_art_classifier(model)

        fgsm = FastGradientMethod(
            estimator=classifier,
            eps=FGSM_EPSILON
        )

        results = evaluate_single_attack(
            model=model,
            classifier=classifier,
            test_loader=test_loader,
            attack_name="FGSM",
            attack_object=fgsm
        )

        results["attack_parameters"] = {
            "epsilon": FGSM_EPSILON
        }

        return results

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# PGD ATTACK
# =============================================================================

def run_pgd_attack(
    model,
    test_loader
):

    """
    Runs PGD adversarial evaluation.
    """

    try:

        classifier = get_art_classifier(model)

        pgd = ProjectedGradientDescent(
            estimator=classifier,
            eps=PGD_CONFIG["eps"],
            eps_step=PGD_CONFIG["alpha"],
            max_iter=PGD_CONFIG["num_steps"]
        )

        results = evaluate_single_attack(
            model=model,
            classifier=classifier,
            test_loader=test_loader,
            attack_name="PGD",
            attack_object=pgd
        )

        results["attack_parameters"] = {
            "epsilon": PGD_CONFIG["eps"],
            "step_size": PGD_CONFIG["alpha"],
            "iterations": PGD_CONFIG["num_steps"]
        }

        return results

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# DEEPFOOL ATTACK
# =============================================================================

def run_deepfool_attack(
    model,
    test_loader
):

    """
    Runs DeepFool adversarial evaluation.
    """

    try:

        classifier = get_art_classifier(model)

        deepfool = DeepFool(
            classifier=classifier,
            max_iter=DEEPFOOL_CONFIG["max_iter"],
            epsilon=DEEPFOOL_CONFIG["overshoot"]
        )

        results = evaluate_single_attack(
            model=model,
            classifier=classifier,
            test_loader=test_loader,
            attack_name="DeepFool",
            attack_object=deepfool
        )

        results["attack_parameters"] = {
            "max_iterations": DEEPFOOL_CONFIG["max_iter"],
            "overshoot": DEEPFOOL_CONFIG["overshoot"]
        }

        return results

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# COMBINED THREAT EVALUATION
# =============================================================================

def combined_threat_evaluation(
    model,
    test_loader
):

    """
    Simulates:
    distribution shift + adversarial attack
    simultaneously.

    This directly supports the integrated
    threat-model claim in the research paper.
    """

    try:

        logging.info(
            "Starting combined threat evaluation"
        )

        classifier = get_art_classifier(model)

        pgd = ProjectedGradientDescent(
            estimator=classifier,
            eps=PGD_CONFIG["eps"],
            eps_step=PGD_CONFIG["alpha"],
            max_iter=PGD_CONFIG["num_steps"]
        )

        correct = 0
        total = 0

        start_time = time.time()

        model.eval()

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # =========================================================
            # SIMULATED DISTRIBUTIONAL SHIFT
            # =========================================================

            drift_noise = torch.randn_like(images) * 0.10

            shifted_images = torch.clamp(
                images + drift_noise,
                0,
                1
            )

            shifted_np = shifted_images.cpu().numpy()

            # =========================================================
            # ADVERSARIAL ATTACK ON DRIFTED DATA
            # =========================================================

            adv_images = pgd.generate(
                x=shifted_np
            )

            adv_tensor = torch.tensor(
                adv_images,
                dtype=torch.float32
            ).to(DEVICE)

            outputs = model(adv_tensor)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        combined_accuracy = correct / total

        runtime = time.time() - start_time

        logging.info(
            "Combined threat evaluation completed"
        )

        return {
            "combined_threat_accuracy": combined_accuracy,
            "runtime_seconds": runtime,
            "pipeline": [
                "Distribution Shift",
                "PGD Attack",
                "Robustness Evaluation"
            ]
        }

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# VISUALIZATION
# =============================================================================

def save_attack_comparison_plot(results_dict):

    """
    Creates high-resolution attack comparison plot.
    """

    try:

        attacks = list(results_dict.keys())

        accuracies = [
            results_dict[a]["accuracy"]
            for a in attacks
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(attacks, accuracies)

        plt.ylabel("Accuracy")

        plt.title(
            "Model Accuracy Under Adversarial Attacks"
        )

        save_path = os.path.join(
            RESULTS_DIR,
            "attack_comparison.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logging.info(
            f"Saved attack comparison plot: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# JSON REPORT EXPORT
# =============================================================================

def save_attack_report(results_dict):

    """
    Saves structured adversarial evaluation report.
    """

    try:

        save_path = os.path.join(
            RESULTS_DIR,
            "adversarial_summary.json"
        )

        serializable_results = {}

        for attack, values in results_dict.items():

            serializable_results[attack] = {
                k: v
                for k, v in values.items()
                if k != "examples"
            }

        with open(save_path, "w") as file:

            json.dump(
                serializable_results,
                file,
                indent=4
            )

        logging.info(
            f"Saved adversarial report: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)
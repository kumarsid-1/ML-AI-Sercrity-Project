import sys
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.config import DEVICE, OUT_DIR
from src.logger import logging
from src.exception import CustomException
from src.utils import save_fig


# ============================================================================
# ATTACK IMPLEMENTATIONS
# ============================================================================

def fgsm_attack(model, img, lbs, eps: float = 0.2):
    """
    Fast Gradient Sign Method (FGSM) - Single-step attack.
    
    Args:
        model: Neural network model
        img: Input images [B, C, H, W]
        lbs: True labels [B]
        eps: Perturbation magnitude
    
    Returns:
        Adversarial images
    """
    try:
        img = img.clone().detach().to(DEVICE)
        lbs = lbs.clone().detach().to(DEVICE)
        img.requires_grad = True
        preds = model(img)
        loss = nn.CrossEntropyLoss()(preds, lbs)
        model.zero_grad()
        loss.backward()
        perturbed = img + eps * img.grad.sign()
        return torch.clamp(perturbed, 0, 1).detach()
    
    except Exception as e:
        raise CustomException(e, sys)


def pgd_attack(model, img, lbs, eps: float = 0.3, alpha: float = 0.01, num_steps: int = 40):
    """
    Projected Gradient Descent (PGD) - Multi-step iterative attack.
    Stronger than FGSM due to iterative refinement and random start.
    
    Args:
        model: Neural network model
        img: Input images [B, C, H, W]
        lbs: True labels [B]
        eps: Maximum perturbation bound (L-infinity)
        alpha: Step size per iteration
        num_steps: Number of attack iterations
    
    Returns:
        Adversarial images
    """
    try:
        img = img.clone().detach().to(DEVICE)
        lbs = lbs.clone().detach().to(DEVICE)

        # Random initialization within epsilon ball (key difference from FGSM)
        adv_img = img.clone().detach()
        adv_img += torch.empty_like(adv_img).uniform_(-eps, eps)
        adv_img = torch.clamp(adv_img, 0, 1).detach()

        for step in range(num_steps):
            adv_img.requires_grad = True
            preds = model(adv_img)
            loss = nn.CrossEntropyLoss()(preds, lbs)
            model.zero_grad()
            loss.backward()

            # Gradient ascent step
            adv_img = adv_img.detach() + alpha * adv_img.grad.sign()

            # Project back into epsilon ball around original image
            delta = torch.clamp(adv_img - img, -eps, eps)
            adv_img = torch.clamp(img + delta, 0, 1).detach()

        return adv_img

    except Exception as e:
        raise CustomException(e, sys)


def deepfool_attack(model, img, lbs, num_classes: int = 10, max_iter: int = 50, overshoot: float = 0.02):
    """
    DeepFool - Finds minimal perturbation to reach decision boundary.
    Exposes model fragility by computing exact distance to nearest class.
    
    Args:
        model: Neural network model
        img: Input images [B, C, H, W]
        lbs: True labels [B] (not directly used, but kept for API consistency)
        num_classes: Number of output classes
        max_iter: Maximum iterations per sample
        overshoot: Small constant to ensure boundary crossing
    
    Returns:
        Adversarial images
    """
    try:
        img = img.clone().detach().to(DEVICE)
        model.eval()

        batch_adv = []

        for idx in range(img.shape[0]):
            x = img[idx:idx+1].clone().detach()
            x_adv = x.clone().detach()

            for iteration in range(max_iter):
                x_adv.requires_grad = True
                output = model(x_adv)
                orig_class = output.argmax(1).item()

                # Compute gradient for original predicted class
                model.zero_grad()
                output[0, orig_class].backward(retain_graph=True)
                grad_orig = x_adv.grad.data.clone()

                # Find minimal perturbation across all other classes
                min_pert = float("inf")
                w_min = None
                f_min = None

                for k in range(num_classes):
                    if k == orig_class:
                        continue
                    
                    x_adv.grad = None
                    x_adv.requires_grad = True
                    out = model(x_adv)
                    model.zero_grad()
                    out[0, k].backward(retain_graph=True)
                    grad_k = x_adv.grad.data.clone()

                    # Compute distance to class k boundary
                    w_k = grad_k - grad_orig
                    f_k = (out[0, k] - out[0, orig_class]).item()
                    pert_k = abs(f_k) / (w_k.norm().item() + 1e-8)

                    if pert_k < min_pert:
                        min_pert = pert_k
                        w_min = w_k
                        f_min = f_k

                # Apply minimal perturbation
                r = (abs(f_min) / (w_min.norm() ** 2 + 1e-8)) * w_min
                x_adv = torch.clamp(x_adv.detach() + (1 + overshoot) * r, 0, 1)

                # Stop if misclassified
                if model(x_adv).argmax(1).item() != orig_class:
                    break

            batch_adv.append(x_adv.detach())

        return torch.cat(batch_adv, dim=0)

    except Exception as e:
        raise CustomException(e, sys)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_single_attack(model, test_loader, attack_name, attack_fn, attack_params, visualize=True):
    """
    Evaluate a single adversarial attack and generate reports.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        attack_name: Name of attack (for logging/saving)
        attack_fn: Attack function (fgsm_attack, pgd_attack, deepfool_attack)
        attack_params: Dictionary of attack parameters
        visualize: Whether to generate visualization plots
    
    Returns:
        Dictionary with clean_acc, adv_acc, accuracy_drop
    """
    logger = logging.getLogger(f"eval_{attack_name}")
    
    try:
        logger.info(f"Starting {attack_name.upper()} evaluation")
        logger.info(f"Parameters: {attack_params}")

        img, lbs = next(iter(test_loader))
        img, lbs = img.to(DEVICE), lbs.to(DEVICE)
        model.eval()

        # Clean accuracy
        logger.info("Evaluating on clean examples")
        with torch.no_grad():
            clean_preds = model(img).argmax(1)
        clean_acc = (clean_preds == lbs).float().mean().item()
        logger.info(f"Clean accuracy: {clean_acc:.5f}")

        # Generate adversarial examples
        logger.info(f"Generating {attack_name} adversarial examples")
        adv_img = attack_fn(model, img, lbs, **attack_params)

        # Adversarial accuracy
        logger.info("Evaluating on adversarial examples")
        with torch.no_grad():
            adv_preds = model(adv_img).argmax(1)
        adv_acc = (adv_preds == lbs).float().mean().item()
        
        acc_drop = clean_acc - adv_acc
        logger.info(f"Adversarial accuracy: {adv_acc:.5f}")
        logger.info(f"Accuracy drop: {acc_drop:.5f}")

        if visualize:
            # Visualization: Clean vs Adversarial examples
            fig, axes = plt.subplots(2, 6, figsize=(13, 5))
            fig.suptitle(f"Clean vs {attack_name.upper()} Adversarial Examples", fontsize=14, fontweight='bold')

            for i in range(6):
                # Clean images
                axes[0][i].imshow(img[i].cpu().squeeze(), cmap="gray")
                axes[0][i].set_title(f"Clean\nPred: {clean_preds[i].item()}", fontsize=10)
                axes[0][i].axis("off")

                # Adversarial images
                axes[1][i].imshow(adv_img[i].cpu().squeeze(), cmap="gray")
                color = "green" if adv_preds[i] == lbs[i] else "red"
                axes[1][i].set_title(f"{attack_name.upper()}\nPred: {adv_preds[i].item()}", 
                                    color=color, fontsize=10)
                axes[1][i].axis("off")

            plt.tight_layout()
            save_fig(fig, f"mnist_{attack_name}_examples.png")
            plt.close(fig)

            # Accuracy bar chart
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(
                ["Clean", f"{attack_name.upper()} Attack"],
                [clean_acc, adv_acc],
                color=["green", "red"],
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"Clean vs {attack_name.upper()} Adversarial Accuracy", fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis="y")

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight='bold'
                )

            save_fig(fig, f"mnist_{attack_name}_acc_bar.png")
            plt.close(fig)

        # Save JSON report
        report = {
            "attack": attack_name.upper(),
            "parameters": attack_params,
            "clean_accuracy": round(clean_acc, 5),
            "adversarial_accuracy": round(adv_acc, 5),
            "accuracy_drop": round(acc_drop, 5),
            "success_rate": round((1 - adv_acc / clean_acc) * 100, 2) if clean_acc > 0 else 0.0
        }
        
        report_path = os.path.join(OUT_DIR, f"{attack_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {report_path}")

        return clean_acc, adv_acc

    except Exception as e:
        raise CustomException(e, sys)


def comprehensive_adversarial_eval(model, test_loader):
    """
    Run all three adversarial attacks (FGSM, PGD, DeepFool) and generate comparison.
    
    Args:
        model: Trained MNIST model
        test_loader: Test data loader
    
    Returns:
        Dictionary with results from all attacks
    """
    logger = logging.getLogger("comprehensive_adversarial_eval")
    
    try:
        logger.info("="*70)
        logger.info("COMPREHENSIVE ADVERSARIAL ROBUSTNESS EVALUATION")
        logger.info("="*70)

        results = {}

        # FGSM Attack
        logger.info("\n[1/3] Running FGSM Attack")
        fgsm_clean, fgsm_adv = evaluate_single_attack(
            model, test_loader, 
            attack_name="fgsm",
            attack_fn=fgsm_attack,
            attack_params={"eps": 0.2}
        )
        results["fgsm"] = {"clean": fgsm_clean, "adversarial": fgsm_adv}

        # PGD Attack
        logger.info("\n[2/3] Running PGD Attack")
        pgd_clean, pgd_adv = evaluate_single_attack(
            model, test_loader,
            attack_name="pgd",
            attack_fn=pgd_attack,
            attack_params={"eps": 0.3, "alpha": 0.01, "num_steps": 40}
        )
        results["pgd"] = {"clean": pgd_clean, "adversarial": pgd_adv}

        # DeepFool Attack
        logger.info("\n[3/3] Running DeepFool Attack")
        deepfool_clean, deepfool_adv = evaluate_single_attack(
            model, test_loader,
            attack_name="deepfool",
            attack_fn=deepfool_attack,
            attack_params={"num_classes": 10, "max_iter": 50, "overshoot": 0.02}
        )
        results["deepfool"] = {"clean": deepfool_clean, "adversarial": deepfool_adv}

        # Comparative Visualization
        logger.info("\nGenerating comparative analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attacks = ["Clean", "FGSM", "PGD", "DeepFool"]
        accuracies = [
            fgsm_clean,  # All clean accuracies should be same, use FGSM's
            fgsm_adv,
            pgd_adv,
            deepfool_adv
        ]
        colors = ["green", "#FF6B6B", "#C0392B", "#8E44AD"]
        
        bars = ax.bar(attacks, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
        ax.set_title("Adversarial Robustness: Attack Comparison", fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight='bold'
            )

        plt.tight_layout()
        save_fig(fig, "mnist_attack_comparison.png")
        plt.close(fig)

        # Summary report
        summary = {
            "clean_accuracy": round(fgsm_clean, 5),
            "attacks": {
                "fgsm": {
                    "adversarial_accuracy": round(fgsm_adv, 5),
                    "accuracy_drop": round(fgsm_clean - fgsm_adv, 5),
                    "attack_success_rate": round((1 - fgsm_adv / fgsm_clean) * 100, 2)
                },
                "pgd": {
                    "adversarial_accuracy": round(pgd_adv, 5),
                    "accuracy_drop": round(pgd_clean - pgd_adv, 5),
                    "attack_success_rate": round((1 - pgd_adv / pgd_clean) * 100, 2)
                },
                "deepfool": {
                    "adversarial_accuracy": round(deepfool_adv, 5),
                    "accuracy_drop": round(deepfool_clean - deepfool_adv, 5),
                    "attack_success_rate": round((1 - deepfool_adv / deepfool_clean) * 100, 2)
                }
            },
            "strongest_attack": min(
                [("fgsm", fgsm_adv), ("pgd", pgd_adv), ("deepfool", deepfool_adv)],
                key=lambda x: x[1]
            )[0].upper()
        }

        summary_path = os.path.join(OUT_DIR, "adversarial_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"\nComprehensive summary saved to {summary_path}")
        logger.info("="*70)
        logger.info("ADVERSARIAL EVALUATION COMPLETE")
        logger.info("="*70)

        return results

    except Exception as e:
        raise CustomException(e, sys)


# Backward compatibility: keep original function signature
def adversarial_eval(model, test_loader, eps: float = 0.2):
    """
    Legacy function - now runs comprehensive evaluation instead of just FGSM.
    Kept for backward compatibility with existing pipeline.
    """
    results = comprehensive_adversarial_eval(model, test_loader)
    # Return FGSM results for backward compatibility
    return results["fgsm"]["clean"], results["fgsm"]["adversarial"]
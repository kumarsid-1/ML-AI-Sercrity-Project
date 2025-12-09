import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.config import DEVICE
from src.logger import logging
from src.exception import CustomException
from src.utils import save_fig


# Fast Gradient Sign Method adverserial attack:
def fgsm_attack(model, img, lbs, eps: float = 0.2):
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


# Evaluating models separately on clean and adversarial examples.
def adversarial_eval(model, test_loader, eps: float = 0.2):
    logger = logging.getLogger("adversarial_eval")
    try:
        logger.info("=" * 60)
        logger.info("Starting ADVERSARIAL ATTACK TESTING (FGSM)")
        logger.info(f"Epsilon: {eps}")
        img, lbs = next(iter(test_loader))
        img, lbs = img.to(DEVICE), lbs.to(DEVICE)

        model.eval()
        logger.info("Evaluating model on clean examples")
        clean_preds = model(img).argmax(1)
        clean_acc = (clean_preds == lbs).float().mean().item()
        logger.info(f"Batch clean accuracy: {clean_acc:.4f}")
        logger.info("Generating FGSM adversarial examples")
        adv_img = fgsm_attack(model, img, lbs, eps=eps)
        logger.info("Evaluating model on adversarial examples")
        adv_preds = model(adv_img).argmax(1)
        adv_acc = (adv_preds == lbs).float().mean().item()

        logger.info(f"Batch adversarial accuracy: {adv_acc:.4f}")
        logger.info(f"Accuracy drop: {clean_acc - adv_acc:.4f}")

        # Visualizing the examples
        fig, axes = plt.subplots(2, 6, figsize=(13, 5))
        fig.suptitle("Clean vs Adversarial Examples")

        for i in range(6):
            axes[0][i].imshow(img[i].cpu().squeeze(), cmap="gray")
            axes[0][i].set_title(f"Clean\nPred: {clean_preds[i].item()}")
            axes[0][i].axis("off")

            axes[1][i].imshow(adv_img[i].cpu().squeeze(), cmap="gray")
            color = "green" if adv_preds[i] == lbs[i] else "red"

            axes[1][i].set_title(
                f"Adversarial\nPred: {adv_preds[i].item()}",
                color=color,
            )
            axes[1][i].axis("off")

        plt.tight_layout()
        save_fig(fig, "mnist_adv_examples.png")
        plt.close(fig)


        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(
            ["Clean", "FGSM Attack"],
            [clean_acc, adv_acc],
            color=["green", "red"],
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Clean vs Adversarial Accuracy")
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )


        save_fig(fig, "mnist_adv_acc_bar.png")
        plt.close(fig)
        return clean_acc, adv_acc


    except Exception as e:
        raise CustomException(e, sys)

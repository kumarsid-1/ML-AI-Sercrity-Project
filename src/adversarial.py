import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import DEVICE
from src.logger import logging
from src.exception import CustomException
from src.utils import save_fig


def fgsm_attack(model, imgs, lbls, eps: float = 0.2):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    """
    try:
        imgs = imgs.clone().detach().to(DEVICE)
        lbls = lbls.clone().detach().to(DEVICE)
        imgs.requires_grad = True

        preds = model(imgs)
        loss = nn.CrossEntropyLoss()(preds, lbls)

        model.zero_grad()
        loss.backward()

        perturbed = imgs + eps * imgs.grad.sign()
        return torch.clamp(perturbed, 0, 1).detach()
    except Exception as e:
        raise CustomException(e, sys)


def adversarial_eval(model, test_loader, eps: float = 0.2):
    """
    Evaluate model on clean and adversarial examples.
    """
    logger = logging.getLogger("adversarial_eval")
    try:
        logger.info("=" * 60)
        logger.info("Starting ADVERSARIAL ATTACK TESTING (FGSM)")
        logger.info(f"Epsilon: {eps}")

        imgs, lbls = next(iter(test_loader))
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        model.eval()
        clean_preds = model(imgs).argmax(1)
        clean_acc = (clean_preds == lbls).float().mean().item()
        logger.info(f"Batch clean accuracy: {clean_acc:.4f}")

        logger.info("Generating FGSM adversarial examples...")
        adv_imgs = fgsm_attack(model, imgs, lbls, eps=eps)
        adv_preds = model(adv_imgs).argmax(1)
        adv_acc = (adv_preds == lbls).float().mean().item()

        logger.info(f"Batch adversarial accuracy: {adv_acc:.4f}")
        logger.info(f"Accuracy drop: {clean_acc - adv_acc:.4f}")

        # Visualize examples
        fig, axes = plt.subplots(2, 6, figsize=(13, 5))
        fig.suptitle("Clean vs Adversarial Examples")

        for i in range(6):
            axes[0][i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
            axes[0][i].set_title(f"Clean\nPred: {clean_preds[i].item()}")
            axes[0][i].axis("off")

            axes[1][i].imshow(adv_imgs[i].cpu().squeeze(), cmap="gray")
            color = "green" if adv_preds[i] == lbls[i] else "red"
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

import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.config import DEVICE
from src.logger import logging
from src.exception import CustomException
from src.utils import save_fig


class TinyCNN(nn.Module):
    """Lightweight CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def train_mnist(epochs: int = 3):
    """
    Train TinyCNN on MNIST (CPU-friendly subset) and save training curves.
    """
    logger = logging.getLogger("train_mnist")
    try:
        logger.info("=" * 60)
        logger.info("Starting MNIST CNN TRAINING")
        logger.info(f"Using device: {DEVICE}")

        transform = transforms.ToTensor()

        logger.info("Downloading MNIST dataset (train/test)...")
        train_data = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )

        subset_size = 10_000
        if len(train_data) > subset_size:
            train_data, _ = torch.utils.data.random_split(
                train_data, [subset_size, len(train_data) - subset_size]
            )
            logger.info(f"Using subset of train data: {subset_size} samples")

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=256, shuffle=False
        )

        model = TinyCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        losses = []

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
            logger.info(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

        # Plot training loss
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, epochs + 1), losses, marker="o")
        ax.set_title("MNIST Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "mnist_loss.png")
        plt.close(fig)

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds = model(imgs.to(DEVICE)).argmax(1)
                correct += (preds == lbls.to(DEVICE)).sum().item()
                total += len(lbls)

        acc = correct / total
        logger.info(f"MNIST test accuracy: {acc:.4f}")

        return model, test_loader, acc

    except Exception as e:
        raise CustomException(e, sys)

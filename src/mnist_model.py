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

# Tiny CNN model for MNIST classification.
class TinyCNN(nn.Module):
    logger = logging.getLogger("TinyCNN")
    def __init__(self):
        self.logger.info("Initializing TinyCNN model")
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

        self.logger.info("TinyCNN model initialized")


    def forward(self, x):
        return self.fc(self.conv(x))

# Train a TinyCNN on MNIST dataset (CPU-friendly subset)
def train_mnist(epochs: int = 3):
    logger = logging.getLogger("train_mnist")

    try:
        logger.info("Initializing CNN TRAINING on MNIST dataset.")
        logger.info(f"Using device: {DEVICE}")

        transform = transforms.ToTensor()
        logger.info("Downloading MNIST dataset")
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
            logger.info(f"Using subset of train data of {subset_size} samples")
    
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=True
        )

        logger.info("Train data loaded")
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=256, shuffle=False
        )

        logger.info("Test data loaded")

        logger.info("Initializing model, optimizer, and criterion")
        model = TinyCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        logger.info("Model, optimizer, and criterion initialized")

        losses = []
        logger.info("Starting training")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for img, lbl in train_loader:
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)

                optimizer.zero_grad()
                preds = model(img)
                loss = criterion(preds, lbl)
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
            for img, lbl in test_loader:
                preds = model(img.to(DEVICE)).argmax(1)
                correct += (preds == lbl.to(DEVICE)).sum().item()
                total += len(lbl)

        acc = correct / total
        logger.info(f"MNIST test accuracy: {acc:.4f}")
        return model, test_loader, acc


    except Exception as e:
        raise CustomException(e, sys)

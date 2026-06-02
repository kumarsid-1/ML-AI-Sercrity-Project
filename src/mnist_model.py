import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.logger import logging
from src.exception import CustomException

from src.config import (
    DEVICE,
    RESULTS_DIR,
    TRAINING_CONFIG,
    ARTIFACTS_DIR
)

# =============================================================================
# CNN MODEL
# =============================================================================

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            9216,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

# =============================================================================
# DATA LOADER
# =============================================================================

def get_data_loaders():

    try:

        logging.info(
            "Loading MNIST dataset"
        )

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False
        )

        logging.info(
            "MNIST dataset loaded successfully"
        )

        return train_loader, test_loader

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SAVE TRAINING CURVE
# =============================================================================

def save_training_curve(epoch_losses):

    try:

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, len(epoch_losses) + 1),
            epoch_losses,
            marker="o"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title(
            "CNN Training Loss Curve",
            fontsize=16
        )

        plt.grid(alpha=0.4)

        save_path = os.path.join(
            RESULTS_DIR,
            "training_loss_curve.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logging.info(
            f"Saved training curve: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(model):

    try:

        save_path = os.path.join(
            ARTIFACTS_DIR,
            "cnn_model.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        logging.info(
            f"Saved trained model: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model():

    try:

        logging.info(
            "Starting CNN training"
        )

        start_time = time.time()

        train_loader, test_loader = get_data_loaders()

        model = CNNModel().to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"]
        )

        epoch_losses = []

        model.train()

        for epoch in range(
            TRAINING_CONFIG["epochs"]
        ):

            running_loss = 0.0

            for images, labels in train_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            epoch_loss = (
                running_loss / len(train_loader)
            )

            epoch_losses.append(
                epoch_loss
            )

            logging.info(
                f"Epoch "
                f"[{epoch+1}/"
                f"{TRAINING_CONFIG['epochs']}] "
                f"Loss: {epoch_loss:.4f}"
            )

        runtime = time.time() - start_time

        logging.info(
            "CNN training completed successfully"
        )

        # =====================================================
        # SAVE MODEL
        # =====================================================

        save_model(model)

        # =====================================================
        # SAVE TRAINING CURVE
        # =====================================================

        save_training_curve(
            epoch_losses
        )

        # =====================================================
        # CLEAN MODEL EVALUATION
        # =====================================================

        clean_accuracy = evaluate_model(
            model,
            test_loader
        )

        # =====================================================
        # TRAINING METADATA
        # =====================================================

        training_metadata = {

            "epochs":
            TRAINING_CONFIG["epochs"],

            "batch_size":
            TRAINING_CONFIG["batch_size"],

            "learning_rate":
            TRAINING_CONFIG["learning_rate"],

            "optimizer":
            TRAINING_CONFIG["optimizer"],

            "device":
            DEVICE,

            "runtime_seconds":
            runtime,

            "clean_accuracy":
            clean_accuracy,

            # REQUIRED FOR ART
            "loss_fn":
            criterion
        }

        return (
            model,
            test_loader,
            training_metadata
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# EVALUATE CLEAN MODEL
# =============================================================================

def evaluate_model(
    model,
    test_loader
):

    try:

        logging.info(
            "Evaluating clean model"
        )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                outputs = model(images)

                preds = outputs.argmax(dim=1)

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = correct / total

        logging.info(
            f"Clean Accuracy: "
            f"{accuracy:.4f}"
        )

        return accuracy

    except Exception as e:

        raise CustomException(e, sys)
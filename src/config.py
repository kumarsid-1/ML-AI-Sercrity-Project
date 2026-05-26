import os
import random
import numpy as np
import torch

from dataclasses import dataclass

# =============================================================================
# PROJECT ROOT PATHS
# =============================================================================

ROOT_DIR = os.getcwd()

ARTIFACTS_DIR = os.path.join(
    ROOT_DIR,
    "artifacts"
)

RESULTS_DIR = os.path.join(
    ROOT_DIR,
    "results"
)

LOG_DIR = os.path.join(
    ROOT_DIR,
    "logs"
)

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data"
)

# =============================================================================
# CREATE REQUIRED DIRECTORIES
# =============================================================================

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

# =============================================================================
# REPRODUCIBILITY CONFIGURATION
# =============================================================================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =============================================================================
# FGSM ATTACK CONFIGURATION
# =============================================================================

FGSM_EPSILON = 0.2

# =============================================================================
# PGD ATTACK CONFIGURATION
# =============================================================================

PGD_CONFIG = {

    # Maximum perturbation magnitude
    "eps": 0.3,

    # Step size
    "alpha": 0.01,

    # Number of iterative optimization steps
    "num_steps": 20,

    # Restrict evaluation size for scalability
    "max_samples": 1000
}

# =============================================================================
# DEEPFOOL ATTACK CONFIGURATION
# =============================================================================

DEEPFOOL_CONFIG = {

    # Total number of classes
    "num_classes": 10,

    # Maximum optimization iterations
    "max_iter": 15,

    # Boundary overshoot factor
    "overshoot": 0.02,

    # Restrict expensive evaluation size
    "max_samples": 1000
}

# =============================================================================
# DRIFT DETECTION CONFIGURATION
# =============================================================================

DRIFT_CONFIG = {

    # Feature index for drift injection
    "feature_index": 0,

    # Human-readable feature name
    "feature_name": "sepal length (cm)",

    # Magnitude of synthetic shift
    "shift_constant": 0.8,

    # Percentage of dataset shifted
    "drift_portion": 0.5,

    # Streaming simulation length
    "stream_length": 600,

    # Point where drift begins
    "drift_start_index": 200
}

# =============================================================================
# MODEL TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {

    # Number of CNN epochs
    "epochs": 3,

    # Mini-batch size
    "batch_size": 64,

    # Optimizer learning rate
    "learning_rate": 1e-3,

    # Optimizer type
    "optimizer": "Adam"
}

# =============================================================================
# SCALABILITY CONFIGURATION
# =============================================================================

SCALABILITY_CONFIG = {

    # Lightweight experimentation mode
    "fast_dev_run": False,

    # Maximum expensive attack samples
    "attack_sample_limit": 1000,

    # Save publication-quality plots
    "save_high_res_plots": True
}

# =============================================================================
# EXPERIMENT TRACKING CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = {

    "project_name": "AI_Security_Pipeline",

    "version": "2.0",

    "author": "Siddharth Kumar",

    "research_focus": [

        "Drift Detection",

        "Adversarial Robustness",

        "Automated ML Testing",

        "Governance Auditing"
    ]
}

# =============================================================================
# RESEARCH METADATA
# =============================================================================

RESEARCH_METADATA = {

    "framework": (
        "PyTorch + ART + River + Giskard"
    ),

    "datasets": [

        "Iris",

        "MNIST"
    ],

    "attacks": [

        "FGSM",

        "PGD",

        "DeepFool"
    ],

    "drift_detectors": [

        "KS-Test",

        "PSI",

        "ADWIN"
    ],

    "hardware": DEVICE,

    "reproducibility_seed": SEED
}

# =============================================================================
# SET GLOBAL RANDOM SEED
# =============================================================================

def set_seed(
    seed: int = SEED
):

    """
    Sets all random seeds
    for full experiment reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class AttackConfig:

    fgsm_eps: float = FGSM_EPSILON

    pgd_eps: float = PGD_CONFIG["eps"]

    pgd_alpha: float = PGD_CONFIG["alpha"]

    pgd_steps: int = PGD_CONFIG["num_steps"]

    pgd_max_samples: int = PGD_CONFIG["max_samples"]

    deepfool_max_iter: int = (
        DEEPFOOL_CONFIG["max_iter"]
    )

    deepfool_overshoot: float = (
        DEEPFOOL_CONFIG["overshoot"]
    )

    deepfool_max_samples: int = (
        DEEPFOOL_CONFIG["max_samples"]
    )

@dataclass
class DriftConfig:

    feature_name: str = (
        DRIFT_CONFIG["feature_name"]
    )

    shift_constant: float = (
        DRIFT_CONFIG["shift_constant"]
    )

    drift_portion: float = (
        DRIFT_CONFIG["drift_portion"]
    )

    drift_start_index: int = (
        DRIFT_CONFIG["drift_start_index"]
    )

@dataclass
class TrainingConfig:

    epochs: int = (
        TRAINING_CONFIG["epochs"]
    )

    batch_size: int = (
        TRAINING_CONFIG["batch_size"]
    )

    learning_rate: float = (
        TRAINING_CONFIG["learning_rate"]
    )

    optimizer: str = (
        TRAINING_CONFIG["optimizer"]
    )

@dataclass
class ScalabilityConfig:

    fast_dev_run: bool = (
        SCALABILITY_CONFIG["fast_dev_run"]
    )

    attack_sample_limit: int = (
        SCALABILITY_CONFIG["attack_sample_limit"]
    )

    save_high_res_plots: bool = (
        SCALABILITY_CONFIG["save_high_res_plots"]
    )
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import SEED
from src.logger import logging
from src.exception import CustomException
from src.utils import adwin_change_detected, compute_psi, save_fig


# Iris classification and drift detection with ADWIN.
def iris_pipeline():
    logger = logging.getLogger("iris_pipeline")
    try:
        logger.info("=" * 50)
        logger.info("Starting IRIS CLASSIFICATION & DRIFT DETECTION")
        iris = load_iris(as_frame=True)
        X: pd.DataFrame = iris.data
        y = iris.target


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingClassifier(random_state=SEED)
        model.fit(X_train_s, y_train)


        preds = model.predict(X_test_s)
        base_acc = accuracy_score(y_test, preds)
        logger.info(f"Baseline accuracy: {base_acc:.4f}")
        logger.info("=" * 50)

        logger.info("Starting drift detection")
        feature = X.columns[0]
        base_vals = X_train[feature].values
        shifted_vals = X_test[feature].values + 0.8
        ks_stat, ks_p = ks_2samp(base_vals, shifted_vals)
        psi_val = compute_psi(base_vals, shifted_vals)

        logger.info(f"KS statistic: {ks_stat:.4f}, p-value: {ks_p:.6f}")
        logger.info(f"PSI: {psi_val:.4f}")


        # Ddistribution plots
        logger.info("Plotting distributions")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(base_vals, bins=20, alpha=0.6, label="Train", color="blue")
        ax.hist(X_test[feature], bins=20, alpha=0.6, label="Test", color="green")
        ax.hist(shifted_vals, bins=20, alpha=0.6, label="Shifted", color="orange")
        ax.set_title(f"Distribution of '{feature}'")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, "iris_feature_dist.png")
        plt.close(fig)
        logger.info("Done")
        logger.info("=" * 50)

        # Drift stream
        logger.info("Generating drift stream")
        rng = np.random.default_rng(SEED)
        base_mean = float(np.mean(base_vals))
        base_std = float(np.std(base_vals) + 1e-3)


        drift_start = 200
        length = 600
        pre = rng.normal(loc=base_mean, scale=base_std, size=drift_start)
        post = rng.normal(
            loc=base_mean + 2.0 * base_std,
            scale=base_std * 1.5,
            size=length - drift_start,
        )


        stream = np.concatenate([pre, post])
        logger.info(f"Drift starts at index {drift_start}, Stream length: {len(stream)}")
        logger.info("=" * 50)
        logger.info("Starting ADWIN drift detection")


        adwin = ADWIN()
        detected = []
        for i, val in enumerate(stream):
            adwin.update(val)
            if adwin_change_detected(adwin):
                detected.append(i)

        logger.info(f"ADWIN detected {len(detected)} drift points")
        if detected:
            logger.info(f"First detection index: {detected[0]}")

        # Plotting stream and detections
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(stream, label="Feature stream", alpha=0.8)
        if detected:
            ax.scatter(
                detected,
                stream[detected],
                s=100,
                color="red",
                label=f"ADWIN Detections ({len(detected)})",
                zorder=5,
                edgecolors="darkred",
                linewidths=2,
                marker="o",
            )


        ax.legend()
        ax.set_title("ADWIN Drift Detection")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Feature Value")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "iris_adwin_stream.png")
        plt.close(fig)
        logger.info("IRIS CLASSIFICATION & DRIFT DETECTION COMPLETED")
        logger.info("=" * 50)


        return {
            "baseline": base_acc,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "psi": psi_val,
            "adwin_first": detected[0] if detected else None,
            "adwin_total": len(detected),
            "adwin_all": detected,
        }


    except Exception as e:
        raise CustomException(e, sys)

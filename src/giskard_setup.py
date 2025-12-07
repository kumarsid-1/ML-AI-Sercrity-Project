import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from giskard import Dataset, Model, Suite, scan
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import OUT_DIR, SEED, set_seed
from src.exception import CustomException
from src.logger import logging


def build_iris_giskard_objects() -> Tuple[Model, Dataset]:
    """
    Train a GradientBoosting model on Iris and wrap it into Giskard Model & Dataset.
    """
    logger = logging.getLogger("giskard_build_iris")
    try:
        set_seed(SEED)
        logger.info("Building Iris model and Giskard Dataset/Model wrappers")

        iris = load_iris(as_frame=True)
        X: pd.DataFrame = iris.data.copy()
        y: pd.Series = iris.target.copy()

        feature_names = list(X.columns)
        target_name = "target"

        # Map numeric labels to string labels for clarity
        label_map = {i: name for i, name in enumerate(iris.target_names)}
        y_str = y.map(label_map)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_str,
            test_size=0.2,
            stratify=y_str,
            random_state=SEED,
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        clf = GradientBoostingClassifier(random_state=SEED)
        clf.fit(X_train_s, y_train)
        logger.info("Iris GradientBoosting model trained for Giskard")

        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            data = df[feature_names]
            data_s = scaler.transform(data)
            return clf.predict_proba(data_s)

        df_full = X.copy()
        df_full[target_name] = y_str

        giskard_dataset = Dataset(
            df=df_full,
            target=target_name,
            name="Iris dataset (GradientBoosting)",
            cat_columns=[],
        )

        classification_labels = list(clf.classes_)
        logger.info(f"Giskard model classification labels: {classification_labels}")

        giskard_model = Model(
            model=prediction_function,
            model_type="classification",
            name="Iris GradientBoosting (Giskard)",
            description="Iris classifier with StandardScaler + GradientBoostingClassifier",
            feature_names=feature_names,
            classification_labels=classification_labels,
        )

        return giskard_model, giskard_dataset

    except Exception as e:
        raise CustomException(e, sys)


def run_iris_scan(save_html: bool = True):
    """
    Run Giskard vulnerability scan on the Iris model & dataset.
    """
    logger = logging.getLogger("giskard_run_scan")
    try:
        logger.info("=" * 60)
        logger.info("Giskard: Iris vulnerability scan starting")

        model, dataset = build_iris_giskard_objects()

        logger.info(f"Dataset shape: {dataset.df.shape}")
        logger.info(f"Dataset target column: {dataset.target}")
        logger.info(f"Feature columns: {model.meta.feature_names}")
        logger.info(f"Model labels: {model.meta.classification_labels}")

        scan_report = scan(model, dataset)
        logger.info("Giskard scan completed")

        if save_html:
            os.makedirs(OUT_DIR, exist_ok=True)
            html_path = os.path.join(OUT_DIR, "iris_giskard_scan.html")
            scan_report.to_html(html_path)
            logger.info(f"Giskard HTML report saved to: {html_path}")

        return scan_report

    except Exception as e:
        raise CustomException(e, sys)


def generate_iris_test_suite(suite_name: str = "Iris security & robustness suite"):
    """
    Run the scan (if needed) and generate a Giskard test suite from it.
    """
    logger = logging.getLogger("giskard_generate_suite")
    try:
        scan_report = run_iris_scan(save_html=True)

        logger.info("Generating Giskard test suite from scan report")
        suite: Suite = scan_report.generate_test_suite(suite_name)

        save_dir = os.path.join(OUT_DIR, "iris_giskard_suite")
        os.makedirs(save_dir, exist_ok=True)
        suite.save(save_dir)

        logger.info(f"Giskard test suite saved in folder: {save_dir}")
        return suite

    except Exception as e:
        raise CustomException(e, sys)


def main():
    """
    CLI entrypoint: python -m src.giskard_setup
    """
    logger = logging.getLogger("giskard_cli")
    try:
        logger.info("Running Giskard setup CLI")
        generate_iris_test_suite()
        logger.info("Giskard setup completed")
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

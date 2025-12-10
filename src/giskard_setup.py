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

# Building the Iris model and the Giskard Model wrappers.
def build_iris_giskard_objects() -> Tuple[Model, Dataset]:
    logging.info("=" * 50)
    logger = logging.getLogger("giskard_build_iris")
    try:
        set_seed(SEED)
        logger.info("Building Iris model and Giskard Model wrappers")
    
        
        iris = load_iris(as_frame=True)
        X: pd.DataFrame = iris.data.copy()
        y: pd.Series = iris.target.copy()
        feature_names = list(X.columns)
        target_name = "target"


        # Mapping numeric features to string features
        label_map = {i: name for i, name in enumerate(iris.target_names)}
        y_str = y.map(label_map)
        logger.info("Iris data loaded")


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_str,
            test_size=0.2,
            stratify=y_str,
            random_state=SEED,
        )
        logger.info("Iris data split into train and test sets")
        
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        logger.info("Iris StandardScaler model trained for Giskard")


        clfs = GradientBoostingClassifier(random_state=SEED)
        clfs.fit(X_train_s, y_train)
        logger.info("Iris GradientBoosting model trained for Giskard")
        

        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            data = df[feature_names]
            data_s = scaler.transform(data)
            return clfs.predict_proba(data_s)


        df_full = X.copy()
        df_full[target_name] = y_str


        giskard_dataset = Dataset(
            df=df_full,
            target=target_name,
            name="Iris dataset (GradientBoosting)",
            cat_columns=[],
        )
        logger.info("Giskard dataset created")


        classification_labels = list(clfs.classes_)
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
        logging.info("Iris model and Giskard Model wrappers built")


    except Exception as e:
        raise CustomException(e, sys)

# Runing the Giskard vulnerability scan on the Iris dataset and model.
def run_iris_scan(save_html: bool = True):
    logger = logging.getLogger("giskard_run_scan")
    try:
        logger.info("Giskard: Starting Iris vulnerability scan")


        model, dataset = build_iris_giskard_objects()
        logger.info("Iris model and Giskard dataset built")
        logger.info(f"Dataset shape: {dataset.df.shape}")
        logger.info(f"Dataset target column: {dataset.target}")
        logger.info(f"Feature columns: {model.meta.feature_names}")
        logger.info(f"Model labels: {model.meta.classification_labels}")


        logger.info("Running Giskard scan")
        scan_report = scan(model, dataset)
        logger.info("Giskard scan completed")
        logger.info("Giskard scan report:")
        if save_html:
            os.makedirs(OUT_DIR, exist_ok=True)
            html_path = os.path.join(OUT_DIR, "iris_giskard_scan.html")
            scan_report.to_html(html_path)
            logger.info(f"Giskard HTML report saved to: {html_path}")
        return scan_report
        logging.info("Giskard scan report generated")


    except Exception as e:
        raise CustomException(e, sys)

# Running the scan (if required) and generating a Giskard test suite from it.
def generate_iris_test_suite(suite_name: str = "Iris security & robustness suite"):
    logger = logging.getLogger("giskard_generate_suite")
    try:
        logger.info("Running Giskard scan and generating test suite")
        scan_report = run_iris_scan(save_html=True)
        suite: Suite = scan_report.generate_test_suite(suite_name)
        logger.info("Giskard test suite generated")


        save_dir = os.path.join(OUT_DIR, "iris_giskard_suite")
        os.makedirs(save_dir, exist_ok=True)
        suite.save(save_dir)


        logger.info(f"Giskard test suite saved in folder: {save_dir}")
        return suite


    except Exception as e:
        raise CustomException(e, sys)

# CLI entrypoint: python -m src.giskard_setup.py (Run in Terminal for project execution)
def main():
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

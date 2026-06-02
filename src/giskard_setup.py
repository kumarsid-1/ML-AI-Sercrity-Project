import os
import sys
import json
import logging
import warnings
import pandas as pd
import giskard

from src.logger import logging as custom_logger
from src.exception import CustomException

from src.config import RESULTS_DIR

# =============================================================================
# SUPPRESS EXTERNAL LIBRARY VERBOSITY
# =============================================================================

warnings.filterwarnings("ignore")

logging.getLogger("giskard").setLevel(logging.ERROR)

logging.getLogger("mlflow").setLevel(logging.ERROR)

# =============================================================================
# GISKARD MODEL WRAPPER
# =============================================================================

def create_giskard_model(
    model,
    prediction_function,
    model_name="MNIST Security Model"
):

    """
    Wraps the trained ML model
    using Giskard's Model API.
    """

    try:

        custom_logger.info(
            "Initializing Giskard model wrapper"
        )

        wrapped_model = giskard.Model(

    model=prediction_function,

    model_type="classification",

    classification_labels=[

        "Digit_0",
        "Digit_1",
        "Digit_2",
        "Digit_3",
        "Digit_4",
        "Digit_5",
        "Digit_6",
        "Digit_7",
        "Digit_8",
        "Digit_9"
    ],

    name=model_name
)

        custom_logger.info(
            "Giskard model wrapper created successfully"
        )

        return wrapped_model

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# GISKARD DATASET WRAPPER
# =============================================================================

def create_giskard_dataset(
    dataframe,
    target_column=None,
    dataset_name="Security Evaluation Dataset"
):

    """
    Converts Pandas DataFrame into
    Giskard Dataset object.
    """

    try:

        custom_logger.info(
            "Creating Giskard dataset wrapper"
        )

        dataset = giskard.Dataset(
            df=dataframe,
            target=target_column,
            name=dataset_name
        )

        custom_logger.info(
            "Giskard dataset created successfully"
        )

        return dataset

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# RUN GISKARD SECURITY SCAN
# =============================================================================

def run_giskard_scan(
    model,
    dataset
):

    """
    Executes automated Giskard scan:
    - bias detection
    - vulnerability detection
    - governance auditing
    - slice-based analysis
    """

    try:

        custom_logger.info(
            "Starting Giskard security scan"
        )

        scan_report = giskard.scan(
            model=model,
            dataset=dataset
        )

        custom_logger.info(
            f"Giskard scan completed | "
            f"Issues detected: {len(scan_report.issues)}"
        )

        return scan_report

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# EXTRACT STRUCTURED ISSUE SUMMARY
# =============================================================================

def extract_issue_summary(
    scan_report
):

    """
    Converts verbose Giskard findings
    into structured JSON-friendly format.
    """

    try:

        issues_summary = []

        for issue in scan_report.issues:

            issue_data = {
                "issue_type": str(type(issue).__name__),
                "description": getattr(
                    issue,
                    "description",
                    "No description available"
                ),
                "severity": getattr(
                    issue,
                    "severity",
                    "Unknown"
                )
            }

            issues_summary.append(issue_data)

        summary = {
            "total_issues_detected": len(issues_summary),
            "issues": issues_summary
        }

        custom_logger.info(
            "Structured Giskard summary generated"
        )

        return summary

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# SAVE GISKARD REPORT
# =============================================================================

def save_giskard_report(
    report_data,
    filename="giskard_summary.json"
):

    """
    Saves summarized governance
    and vulnerability findings.
    """

    try:

        save_path = os.path.join(
            RESULTS_DIR,
            filename
        )

        with open(save_path, "w") as file:

            json.dump(
                report_data,
                file,
                indent=4
            )

        custom_logger.info(
            f"Giskard summary saved: {save_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)

# =============================================================================
# COMPLETE GISKARD PIPELINE
# =============================================================================

def run_complete_giskard_pipeline(
    model,
    prediction_function,
    dataframe,
    target_column=None
):

    """
    Full governance pipeline:
    1. Wrap model
    2. Wrap dataset
    3. Run scan
    4. Extract findings
    5. Save structured report
    """

    try:

        custom_logger.info(
            "Starting complete Giskard pipeline"
        )

        wrapped_model = create_giskard_model(
            model=model,
            prediction_function=prediction_function
        )

        wrapped_dataset = create_giskard_dataset(
            dataframe=dataframe,
            target_column=target_column
        )

        scan_report = run_giskard_scan(
            model=wrapped_model,
            dataset=wrapped_dataset
        )

        structured_summary = extract_issue_summary(
            scan_report
        )

        save_giskard_report(
            structured_summary
        )

        custom_logger.info(
            "Complete Giskard pipeline finished successfully"
        )

        return structured_summary

    except Exception as e:

        raise CustomException(e, sys)
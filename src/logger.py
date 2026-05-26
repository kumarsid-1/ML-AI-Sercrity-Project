import os
import logging
from datetime import datetime

# =============================================================================
# LOG DIRECTORY SETUP
# =============================================================================

LOG_DIR = os.path.join(
    os.getcwd(),
    "logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

# =============================================================================
# LOG FILE CONFIGURATION
# =============================================================================

LOG_FILE_NAME = (
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
)

LOG_FILE_PATH = os.path.join(
    LOG_DIR,
    LOG_FILE_NAME
)

# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format=(
        "[ %(asctime)s ] "
        "%(lineno)d "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    )
)

# =============================================================================
# CONSOLE HANDLER
# =============================================================================

console_handler = logging.StreamHandler()

console_handler.setLevel(logging.INFO)

console_formatter = logging.Formatter(
    (
        "[ %(asctime)s ] "
        "%(lineno)d "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    )
)

console_handler.setFormatter(
    console_formatter
)

# =============================================================================
# ROOT LOGGER
# =============================================================================

root_logger = logging.getLogger()

if not root_logger.handlers:

    root_logger.addHandler(
        console_handler
    )

else:

    root_logger.handlers.clear()

    root_logger.addHandler(
        console_handler
    )

# =============================================================================
# FILE HANDLER
# =============================================================================

file_handler = logging.FileHandler(
    LOG_FILE_PATH
)

file_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter(
    (
        "[ %(asctime)s ] "
        "%(lineno)d "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    )
)

file_handler.setFormatter(
    file_formatter
)

root_logger.addHandler(
    file_handler
)

root_logger.setLevel(
    logging.INFO
)

# =============================================================================
# REDUCE EXTERNAL LIBRARY VERBOSITY
# =============================================================================

logging.getLogger(
    "matplotlib"
).setLevel(logging.WARNING)

logging.getLogger(
    "PIL"
).setLevel(logging.WARNING)

logging.getLogger(
    "giskard"
).setLevel(logging.ERROR)

logging.getLogger(
    "mlflow"
).setLevel(logging.ERROR)

# =============================================================================
# STARTUP LOG
# =============================================================================

logging.info(
    "Centralized logging initialized successfully"
)

logging.info(
    f"Log file created at: {LOG_FILE_PATH}"
)
import logging
import os
from datetime import datetime

# Create logs directory with timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Basic configuration: logs go to file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
    level=logging.INFO,
)

# If you also want logs on console, uncomment this:
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_formatter = logging.Formatter(
#     "[ %(asctime)s ] %(levelname)s [%(name)s:%(lineno)d] - %(message)s"
# )
# console_handler.setFormatter(console_formatter)
# logging.getLogger().addHandler(console_handler)

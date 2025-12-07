import sys
from pathlib import Path

# Ensure project root is on the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Build a readable error message with file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return (
        f"Error occurred in python script [{file_name}] "
        f"at line [{line_no}]: {str(error)}"
    )


class CustomException(Exception):
    """
    Custom exception that logs detailed error info.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

import sys
from src.logger import logging


# =============================================================
# CUSTOM ERROR MESSAGE
# =============================================================

def error_message_detail(error, error_detail):

    """
    Generates detailed error message with:
    - file name
    - line number
    - original exception
    """

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"\nError occurred in python script: "
        f"[{file_name}] "
        f"\nLine number: [{exc_tb.tb_lineno}] "
        f"\nError message: [{str(error)}]"
    )

    return error_message


# =============================================================
# CUSTOM EXCEPTION CLASS
# =============================================================

class CustomException(Exception):

    """
    Custom exception class for centralized
    project error handling.
    """

    def __init__(self, error_message, error_detail):

        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

        logging.error(self.error_message)

    def __str__(self):

        return self.error_message
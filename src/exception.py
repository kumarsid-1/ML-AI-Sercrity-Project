import sys
import traceback

from src.logger import logging

# =============================================================================
# ERROR MESSAGE GENERATOR
# =============================================================================

def error_message_detail(
    error,
    error_detail: sys
):

    """
    Generates detailed custom error messages
    including:
    - file name
    - line number
    - actual exception message
    """

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"\nError occurred in Python script:\n"
        f"File Name      : {file_name}\n"
        f"Line Number    : {exc_tb.tb_lineno}\n"
        f"Error Message  : {str(error)}\n"
    )

    return error_message

# =============================================================================
# CUSTOM EXCEPTION CLASS
# =============================================================================

class CustomException(Exception):

    """
    Custom exception class for:
    - structured debugging
    - centralized error tracking
    - cleaner production logging
    - research reproducibility
    """

    def _init_(
        self,
        error_message,
        error_detail: sys
    ):

        super()._init_(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

        # =============================================================
        # LOG FULL TRACEBACK
        # =============================================================

        logging.error(
            "\n========== EXCEPTION TRACEBACK ==========\n"
        )

        logging.error(traceback.format_exc())

        logging.error(
            "=========================================\n"
        )

    def _str_(self):

        return self.error_message

# =============================================================================
# OPTIONAL HELPER DECORATOR
# =============================================================================

def exception_handler(func):

    """
    Optional reusable decorator for:
    automatic exception wrapping.

    Example:
        @exception_handler
        def train():
            ...
    """

    def wrapper(*args, **kwargs):

        try:

            return func(*args, **kwargs)

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    return wrapper
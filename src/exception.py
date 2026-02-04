import sys
from src.logger import logging

import sys
import traceback

class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error(error, error_detail: sys):
        _, _, tb = error_detail.exc_info()
        file_name = tb.tb_frame.f_code.co_filename
        line_no = tb.tb_lineno

        return f"Error occurred in script [{file_name}] at line [{line_no}] : {str(error)}"

    def __str__(self):
        return self.error_message


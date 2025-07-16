import sys
from src.logger import logging

def error_message_details(error, error_details:sys):
    """
    Returns a formatted error message with details.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        """
        Custom exception class that captures error details.
        """
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)
    def __str__(self):
        """
        Returns the string representation of the error message.
        """
        return self.error_message



if __name__ == "__main__":
    try:
        1 / 0  # Example error
    except Exception as e:
        raise CustomException(e, sys) from e  # Raise custom exception with details
        # This will trigger the custom exception and print the error message
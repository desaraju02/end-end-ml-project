import os
import sys
import joblib
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves an object to a file using joblib.

    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The object to be saved.

    Raises:
    CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads an object from a file using joblib.

    Parameters:
    file_path (str): The path from where the object will be loaded.

    Returns:
    The loaded object.

    Raises:
    CustomException: If there is an error during the loading process.
    """
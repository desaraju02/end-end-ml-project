import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
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

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            logging.info(f"{model_name} - Train R2 Score: {train_model_score}, Test R2 Score: {test_model_score}")

        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from dataclasses import dataclass
from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:    
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                # 'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            # best_model_score = max(sorted(model_report.values(), key=lambda x: x['R2_Score'])['R2_Score'])
            # best_model_name = max(model_report, key=lambda x: model_report[x]['R2_Score'])
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                                   
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")
            
           
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model training completed successfully")
            predicted_data = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted_data)
            logging.info(f"R2 Score of the best model on test data: {r2_square}")
            return best_model_name, r2_square
        except Exception as e:
            raise CustomException(e, sys)
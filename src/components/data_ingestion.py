import os
import sys
# from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import save_object, load_object

@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv("data/stud.csv")
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and Test split completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test data saved to artifacts folder")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
            logging.error("Error occurred during data ingestion")
            logging.info("Data Ingestion completed")



if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(
        train_path=obj.ingestion_config.train_data_path,
        test_path=obj.ingestion_config.test_data_path
    )
    model_trainer = ModelTrainer()
    best_model_name, best_model_score = model_trainer.initiate_model_trainer(
        train_array=train_arr,
        test_array=test_arr
    )
    logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
    logging.info("Preprocessor object created successfully")
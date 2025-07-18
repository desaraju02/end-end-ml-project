import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException   
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # "gender","race_ethnicity","parental_level_of_education","lunch",
            # "test_preparation_course","math_score","reading_score","writing_score"
            numerical_features = ['writing_score', 'reading_score']  # Replace with actual numerical features
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education','lunch','test_preparation_course']  # Replace with actual categorical features

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))  # Scaling after one-hot encoding
            ])

            logging.info("Categorical Columns: %s", categorical_features)
            logging.info("Numerical Columns: %s", numerical_features)

            logging.info("Numerical and categorical transformers created successfully.")
            # Create a preprocessor that applies the transformations to the respective features
            logging.info("Creating preprocessor object.")
            # Ensure the features are correctly defined
            if not numerical_features or not categorical_features:
                raise ValueError("Numerical and categorical features must be defined.")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
            logging.info("Data transformation object creation failed.") 


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated.")
            data_transformation = DataTransformation()
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")
            logging.info("Train Data Shape: %s", train_df.shape)
            logging.info("Test Data Shape: %s", test_df.shape)
            preprocessing_obj = data_transformation.get_data_transformer_object()
            logging.info("Preprocessing object created successfully.")
            target_column = 'math_score'  # Replace with your target column name
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the data.")
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info("Input features and target feature separated successfully.")
            # Fit the preprocessor on the training data and transform both train and test data

            input_features_train_transformed = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_transformed = preprocessing_obj.transform(input_features_test_df)
            train_arr = np.c_[input_features_train_transformed, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_transformed, np.array(target_feature_test_df)]
            logging.info("Data transformation applied to train and test data.")

            logging.info("Data transformation applied successfully.")
            # Save the preprocessor object to a file
            save_object (
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            logging.info("Data transformation initiation failed.")  

            
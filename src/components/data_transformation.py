import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X):
        """
        This function is responsible for creating and returning a ColumnTransformer
        with OneHotEncoder for categorical features and StandardScaler for numerical features
        """
        try:
            # Ensure no string dtypes exist before processing
            for col in X.columns:
                if hasattr(X[col].dtype, 'name') and 'string' in str(X[col].dtype):
                    X[col] = X[col].astype('object')
            
            # Separate numerical and categorical features
            num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_features = X.select_dtypes(include=['object']).columns.tolist()

            logging.info(f"Numerical columns: {num_features}")
            logging.info(f"Categorical columns: {cat_features}")
            
            num_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())

                ] )
            cat_pipeline= Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                        ('scaler', StandardScaler(with_mean=False))
                    ]
                )
            logging.info("Numerical and categorical pipelines created")

            # Create transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('cat_pipeline', cat_pipeline, cat_features),
                    ('num_pipeline', num_pipeline, num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_column='math score'):
        """
        Read train and test data, apply transformations, and save preprocessor
        """
        try:
            logging.info("Data Transformation started")

            # Read train and test data
            train_df = pd.read_csv(train_path, dtype=str).infer_objects()
            test_df = pd.read_csv(test_path, dtype=str).infer_objects()

            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Separate features and target
            X_train = train_df.drop('math score', axis=1)
            y_train = pd.to_numeric(train_df['math score'], errors='coerce')

            X_test = test_df.drop('math score', axis=1)
            y_test = pd.to_numeric(test_df['math score'], errors='coerce')
            
            # Convert all string dtypes to object for sklearn compatibility
            X_train = X_train.astype(str).apply(lambda x: x.astype('object') if x.dtype == 'object' else x)
            X_test = X_test.astype(str).apply(lambda x: x.astype('object') if x.dtype == 'object' else x)

            # Get preprocessor object
            preprocessor = self.get_data_transformer_object(X_train)

            # Fit and transform train data
            X_train_transformed = preprocessor.fit_transform(X_train)
            # Transform test data (fit only on train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            # Save preprocessor object
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor object saved")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Initiate data ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Initiate data transformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")
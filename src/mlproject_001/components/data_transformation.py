import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging
from src.mlproject_001.components.data_ingestion import dataIngestion
from src.mlproject_001.utils import save_object

import os





@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()

    def get_data_transformer_obj(self, train_data_path):
        """
        this function is responsible for data transformation
        """
        try:
            ## just to extract columns & dtypes of the data
            df = pd.read_csv(train_data_path)

            X = df.drop(columns=['math score'],axis=1)
            numerical_columns = X.select_dtypes(exclude="object").columns
            categorical_columns = X.select_dtypes(include="object").columns

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading train & test data file")

            preprocessing_obj = self.get_data_transformer_obj(train_data_path=train_data_path)

            target_column = "math score"
            
            # divide data into dv & idv
            X_train = train_df.drop(columns=[target_column],axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column],axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                X_train, np.array(y_train)
            ]
            test_arr = np.c_[
                X_test, np.array(y_test)
            ]

            logging.info("Saved preprocessing object")

            save_object(
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
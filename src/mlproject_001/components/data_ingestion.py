# MySQl --> data --> train test split --> dataset

import os
import sys
import numpy as np
import pandas as pd

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging
from src.mlproject_001.utils import read_sql_data

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class dataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')


class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ## reading the data
            df = read_sql_data()
            logging.info("Reading completed from mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=101)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

import sys
from src.mlproject_001.logger import logging
from src.mlproject_001.exception import CustomException
from src.mlproject_001.components.data_ingestion import dataIngestion
from src.mlproject_001.components.data_ingestion import dataIngestionConfig


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config = dataIngestionConfig()
        data_ingestion = dataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)
import sys
from src.mlproject_001.logger import logging
from src.mlproject_001.exception import CustomException
from src.mlproject_001.components.data_ingestion import dataIngestion
from src.mlproject_001.components.data_transformation import dataTransformationConfig, dataTransformation


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config = dataIngestionConfig()
        data_ingestion = dataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = dataTransformation()
        data_transformation.initiate_data_transformation(train_data_path=train_data_path, test_data_path=test_data_path)
        
    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)
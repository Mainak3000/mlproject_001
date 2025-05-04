import sys
from src.mlproject_001.logger import logging
from src.mlproject_001.exception import CustomException
from src.mlproject_001.components.data_ingestion import dataIngestion
from src.mlproject_001.components.data_transformation import dataTransformationConfig, dataTransformation
from src.mlproject_001.components.model_trainer import modelTrainerConfig, modelTrainer

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config = dataIngestionConfig()
        data_ingestion = dataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # data transformation
        data_transformation = dataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # model training
        model_trainer = modelTrainer()
        model_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(model_r2_score)

    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)
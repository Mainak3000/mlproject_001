from src.mlproject_001.logger import logging
from src.mlproject_001.exception import CustomException
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)
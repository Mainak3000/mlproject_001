import os
import sys
import numpy as np
import pandas as pd

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging

from dotenv import load_dotenv
import redshift_connector

import pickle
import joblib

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        conn_redshift = redshift_connector.connect (
                            host=host,
                            database=db,
                            user=user,
                            password=password
                        )


        logging.info("Connection established: %s", conn_redshift)
        
        df = pd.read_sql("""select * from adf0793.students""", conn_redshift)
        print(df.head())
        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
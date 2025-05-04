import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging

from dotenv import load_dotenv
import redshift_connector

import pickle
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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
        print("Shape of the data", df.shape)
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

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            grid = GridSearchCV(model, param, cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)        
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            test_score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
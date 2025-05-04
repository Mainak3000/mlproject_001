import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging
from src.mlproject_001.utils import evaluate_models, save_object

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "final_model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_tainer_config = modelTrainerConfig()
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split dv and idv from data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2', None],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':[None,'sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params
            )
            
            ## to get best model score from dict
            best_score = max(sorted(model_report.values()))
            
            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]
            best_model = models[best_model_name]

            print(f"best model is : {best_model_name} & parameters : {best_model.get_params()}")
            
            mlflow.set_registry_uri("https://dagshub.com/Mainak3000/mlproject_001.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_model.get_params())

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)

                # Model registry does not work with file store
                input_example = X_test[:1]  # one row of test input
                signature = infer_signature(X_test, best_model.predict(X_test))

                # Log model with signature and input example
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        registered_model_name=best_model_name,
                        input_example=input_example,
                        signature=signature
                    )
                else:
                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        input_example=input_example,
                        signature=signature
                    )

            if best_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both train & test dataset")

            data_arr = np.vstack((train_arr, test_arr))  # Stack rows vertically

            X = data_arr[:, :-1]
            y = data_arr[:, -1]

            best_model.fit(X,y)

            save_object(
                file_path=self.model_tainer_config.trained_model_file_path,
                obj=best_model
            )

            pred = best_model.predict(X)

            r2_square = r2_score(y, pred)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
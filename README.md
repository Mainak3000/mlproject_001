## End to end ML project

This is an end-to-end machine learning project

It includes custom logger, custom exception handler, custom data ingestion, data trainer and data predictor through Flask app

Model data is tracked by DVC and Model runs are tracked by MLflow.

### Running instructuctions

 1. git clone https://github.com/Mainak3000/mlproject_001.git
 2. cd mlproject_001
 3. run conda create -p project_env python==3.9 -y
 4. run pip install -r requirements.txt
 5. run python template.py
 6. To create train data, test data, final model and data preprocessor, run python src/mlproject_001/pipelines/training_pipeline.py 
 7. run python app.py

MLFLOW_TRACKING_URI=https://dagshub.com/Mainak3000/mlproject_001.mlflow \
MLFLOW_TRACKING_USERNAME=Mainak3000 \
MLFLOW_TRACKING_PASSWORD=xxxx \
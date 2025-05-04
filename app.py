from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.mlproject_001.pipelines.prediction_pipeline import customData, predictPipeline

from src.mlproject_001.logger import logging


app = Flask(__name__)

logging.info("Flask app started")

# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data = customData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = predictPipeline()
        result = predict_pipeline.predict(pred_df)

        logging.info("Predictions returned to web-page")
        
        return render_template("home.html", result=result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
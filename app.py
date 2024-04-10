from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
application = Flask(__name__)

app = application
spark = SparkSession.builder.master('local').appName('sparkProject').getOrCreate()
## Route for a home page

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = request.form.get('age'),
            bmi = float(request.form.get('bmi')),
            children = request.form.get('children'),
            sex = request.form.get('sex'),
            smoker = request.form.get('smoker'),
            region = request.form.get('region'),
        )
        pred_df = data.get_data_as_data_frame(spark=spark)
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
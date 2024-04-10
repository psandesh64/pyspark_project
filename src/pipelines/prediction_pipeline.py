import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from pyspark.ml import PipelineModel
from pyspark.ml.regression import DecisionTreeRegressionModel
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model_rf')
            preprocessor_path = os.path.join('artifacts','preprocessor_rf')
            preprocessor= PipelineModel.load(preprocessor_path)
            data_scaled = preprocessor.transform(features)

            # Extract the features column
            features_vector = data_scaled.select('features')
            # Load the model
            model = DecisionTreeRegressionModel.load(model_path)
            # Predict using the model
            preds = model.transform(features_vector)
            return preds.select('prediction').first()
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self,
        age:int,
        bmi:float,
        children: int,
        sex: str,
        smoker: str,
        region: str):
    
        self.age = age
        self.bmi = bmi
        self.children = children
        self.sex = sex
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self,spark):
        try:
            schema = StructType([
                StructField("age", IntegerType(), True),
                StructField("bmi", FloatType(), True),
                StructField("children", IntegerType(), True),
                StructField("sex", StringType(), True),
                StructField("smoker", StringType(), True),
                StructField("region", StringType(), True)
            ])
            data = [(int(self.age), self.bmi, int(self.children), self.sex, self.smoker, self.region)]
            return spark.createDataFrame(data, schema)
        
        except Exception as e:
            raise CustomException(e,sys)
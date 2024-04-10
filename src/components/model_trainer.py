import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from pyspark.ml.regression import (
    RandomForestRegressor,
    DecisionTreeRegressor,
    GBTRegressor,
    LinearRegression
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator



@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model_rf")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_transformed_df,test_transformed_df):
        try:
            logging.info("Split training and test input data")
    
            # Convert RDD to DataFrame

            models = {
                "Random Forest": RandomForestRegressor(featuresCol="features",labelCol="charges"),
                "Decision Tree": DecisionTreeRegressor(featuresCol="features",labelCol="charges"),
                "Gradient Boosting": GBTRegressor(featuresCol="features",labelCol="charges"),
                "Linear Regression": LinearRegression(featuresCol="features",labelCol="charges",regParam=0.01, maxIter=100)
            }
            paramGrid = {
                "Random Forest": ParamGridBuilder().addGrid(RandomForestRegressor.numTrees, [10, 20, 30]).build(),
                "Decision Tree": ParamGridBuilder().addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15]).build(),
                "Gradient Boosting": ParamGridBuilder().addGrid(GBTRegressor.maxDepth, [2, 4, 6]).build(),
                "Linear Regression": ParamGridBuilder().build()
            }

            evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="charges",metricName="r2")

            ## Train and evaluate using cross-validation
            model_report = {}
            for model_name, model in models.items():
                param_grid = paramGrid[model_name]
                crossval = CrossValidator(estimator=model,
                                        estimatorParamMaps=param_grid,
                                        evaluator=evaluator,
                                        numFolds=5)
                cv_model = crossval.fit(train_transformed_df)
                predictions = cv_model.transform(test_transformed_df)
                r2_score = evaluator.evaluate(predictions)
                model_report[model_name] = r2_score

            # Retrieve the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            print(model_report[best_model_name],best_model_name)  

            regressor = best_model.fit(train_transformed_df)
            file_path = self.model_trainer_config.train_model_file_path
            regressor.write().overwrite().save(file_path)

            return model_report[best_model_name]

        except Exception as e:
            raise CustomException(e,sys)
        

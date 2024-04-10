import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,StandardScaler,Imputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor_rf')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.spark = SparkSession.builder.master('local').appName('sparkProject').getOrCreate()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = [
                "age",
                "bmi",
                "children",
                # "charges",                      # remove the charges column since it is input label
                ]
            categorical_columns = [
                "sex",
                "smoker",
                "region",
            ]
            
            imputers_num = [Imputer(inputCol=column, outputCol=column + "_imputed", strategy="mean") for column in numerical_columns]
            assembler_num = VectorAssembler(inputCols=[column + "_imputed" for column in numerical_columns], outputCol="numerical_features")
            scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features", withStd=True, withMean=True)
            indexers = [StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="keep") for column in categorical_columns]
            encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_encoded") for column in categorical_columns]
            assembler_cat = VectorAssembler(inputCols=[column + "_encoded" for column in categorical_columns], outputCol="categorical_features")
            assembler = VectorAssembler(inputCols=["scaled_numerical_features", "categorical_features"], outputCol="features")
            pipeline = Pipeline(stages=imputers_num + [assembler_num] + [scaler] + indexers + encoders + [assembler_cat] + [assembler])

            return pipeline
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = self.spark.read.csv(train_path, header=True, inferSchema=True)
            test_df = self.spark.read.csv(test_path, header=True, inferSchema=True)
            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Applying Preprocessing on training and test dataframe")

            target_column_name="charges"
            # Drop the target column from the train data
            input_features_train_df = train_df.select([col for col in train_df.columns if col != target_column_name])

            preprocessing_model = preprocessing_obj.fit(input_features_train_df)
            input_features_train_arry = preprocessing_model.transform(train_df)
            input_features_test_arry = preprocessing_model.transform(test_df)
            print(input_features_train_arry.printSchema())
            
            train_transformed = input_features_train_arry.select("features", target_column_name)
            test_transformed = input_features_test_arry.select("features", target_column_name)
            train_transformed.show()

            file_path = self.data_transformation_config.preprocessor_obj_file_path
            preprocessing_model.write().overwrite().save(file_path)

            return (
                train_transformed,
                test_transformed
            )

        except Exception as e:
            raise CustomException(sys,e)
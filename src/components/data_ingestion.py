import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from pyspark.sql import SparkSession

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
        self.spark = SparkSession.builder.master('local').appName('sparkProject').getOrCreate()

    def initiate_data_ingestion(self):
        try:
            # Reading data from CSV
            df = self.spark.read.csv("artifacts/medical_insurance.csv", header=True, inferSchema=True)
            
            # Saving raw data to CSV
            df.toPandas().to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Splitting data into train and test sets
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

            # Saving train and test sets to CSV
            train_df.toPandas().to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.toPandas().to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
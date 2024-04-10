from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        print(data_ingestion.spark,'this is the spark session.')

        data_transformation = DataTransformation()
        train_transformed_df, test_transformed_df = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_transformed_df,test_transformed_df)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

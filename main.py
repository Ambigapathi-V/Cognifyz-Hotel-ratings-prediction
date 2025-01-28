from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Hotel_Ratings_prediction.pipeline.data_preparation_pipeline import DataPreparationTrainingPipeline
import sys

STAGE_NAME = "Data Ingestion Stage"

try: 
        pipeline = DataIngestionTrainingPipeline()
        logging.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_ingestion()
        logging.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        logging.info("-----------------------------------------")
        
except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(e, sys)

STAGE_NAME = "Data Preparation Stage"

try: 
        pipeline = DataPreparationTrainingPipeline()
        logging.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_preparation_ingestion()
        logging.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        logging.info("-----------------------------------------")
        
except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(e, sys)
from src.Hotel_Ratings_prediction.logging import logger
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Hotel_Ratings_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
import sys

STAGE_NAME = "Data Ingestion Stage"

try: 
        pipeline = DataIngestionTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_data_ingestion()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        logger.info("-----------------------------------------")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise Exception(e, sys)
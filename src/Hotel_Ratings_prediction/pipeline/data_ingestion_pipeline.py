from src.Hotel_Ratings_prediction.config.configuration import ConfigurationManager
from src.Hotel_Ratings_prediction.components.data_ingestion import DataIngestion
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
import os
import sys

STAGE_NAME = 'Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self):
        
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            df = data_ingestion.load_data()
            data_ingestion.save_df(data_ingestion_config.output_path,df=df)
            
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            return Exception(e,sys)
        



    
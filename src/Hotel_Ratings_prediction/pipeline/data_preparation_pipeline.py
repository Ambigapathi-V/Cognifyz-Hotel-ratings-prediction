from src.Hotel_Ratings_prediction.config.configuration import ConfigurationManager
from src.Hotel_Ratings_prediction.components.data_preparation import DataPreparation
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
import os
import sys

STAGE_NAME = 'Data Preparation Stage'

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_preparation_ingestion(self):
        
        try:
            config = ConfigurationManager()
            data_preparation_config = config.get_data_preparation_config()
            data_preparation = DataPreparation(data_preparation_config)
            data_preparation.preprocessing()
            
        except Exception as e:
            logging.error(f"Error occurred during data Preparation: {e}")
            return Exception(e,sys)
        



    
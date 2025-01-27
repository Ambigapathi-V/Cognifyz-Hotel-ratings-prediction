import os
import sys
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.utils.common import create_directories, load_df
from src.Hotel_Ratings_prediction.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.input_path = config.input_path
        self.output_path = config.output_path

    def load_data(self,):
        try:
            # Log the input path to verify the correct path is passed
            logging.info(f'Loading data from {self.input_path}')
            
            if not os.path.exists(self.input_path):
                logging.error(f"Input path does not exist: {self.input_path}")
                raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
                
            # Load data from the specified input path
            df = load_df(self.input_path)
            
            return df
        except Exception as e:
            logging.error(f'Error loading data: {str(e)}')
            raise e

    def save_df(self, path, df):
        
        try:
            logging.info(f'Saving data to {path}')
            create_directories([os.path.dirname(path)])
            df.to_csv(path, index=False)
            logging.info(f'Saved data to {path}')
            
        except Exception as e:
            logging.error(f'Error saving data: {str(e)}')
            raise e

from src.Hotel_Ratings_prediction.config.configuration import ConfigurationManager
from src.Hotel_Ratings_prediction.components.model_trainer import ModelTrainer
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
import os
import sys

STAGE_NAME = 'Model Trainer Stage'

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_training(self):
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)
            model_trainer.build_model()         
        except Exception as e:
            logging.error(f"Error occurred during data Preparation: {e}")
            return Exception(e,sys)
        



    
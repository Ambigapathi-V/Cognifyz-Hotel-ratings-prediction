from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.constants import *
from src.Hotel_Ratings_prediction.utils.common import read_yaml, create_directories
from src.Hotel_Ratings_prediction.entity.config_entity import (DataIngestionConfig,DataPreparationConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH,
                 schema_file_path=SCHEMA_FILE_PATH):
        # Ensure that the paths are converted to Path objects
        self.config_file_path = Path(config_file_path)
        self.params_file_path = Path(params_file_path)
        self.schema_file_path = Path(schema_file_path)

        # Reading YAML files with the proper paths
        self.config = read_yaml(self.config_file_path)
        self.params = read_yaml(self.params_file_path)
        self.schema = read_yaml(self.schema_file_path)
        
        # Creating necessary directories
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        # Ensure that the paths are converted to Path objects and directories are created
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            input_path=Path(config.input_path),
            output_path=Path(config.output_path),
        )
        
        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        
        # Ensure that the paths are converted to Path objects and directories are created
        create_directories([config.root_dir])
        
        data_preparation_config = DataPreparationConfig(
            root_dir=config.root_dir,
            input_path=Path(config.input_path),
            output_path=Path(config.output_path),
            preprocessor=Path(config.preprocessor),
            target_column = config.target_column,
            X_train_transformed_df = Path(config.X_train_transformed_df),
            Y_train_transformed_df = Path(config.Y_train_transformed_df),
            X_test_transformed_df = Path(config.X_test_transformed_df),
            Y_test_transformed_df = Path(config.Y_test_transformed_df)
        )
        
        return data_preparation_config
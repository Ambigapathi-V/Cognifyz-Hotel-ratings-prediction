from dataclasses import dataclass
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from pathlib import Path

@dataclass
class DataIngestionConfig:
    input_path: Path
    output_path: Path
    root_dir: Path
    
@dataclass
class DataPreparationConfig:
    root_dir: Path
    input_path: Path
    output_path: Path
    preprocessor: Path
    target_column: str
    X_train_transformed_df: Path  # Ensure the naming matches the YAML file
    Y_train_transformed_df: Path  # Update to match YAML naming
    X_test_transformed_df: Path   # Ensure naming matches YAML
    Y_test_transformed_df: Path               # Update to match YAML naming

    
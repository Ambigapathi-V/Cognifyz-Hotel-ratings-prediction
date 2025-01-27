from dataclasses import dataclass
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from pathlib import Path

@dataclass
class DataIngestionConfig:
    input_path: Path
    output_path: Path
    root_dir: Path
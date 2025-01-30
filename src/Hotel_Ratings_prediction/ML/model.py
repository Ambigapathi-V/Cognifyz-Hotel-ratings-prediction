import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

class ModelArchitecture:
    """
    A class to define and manage the training of machine learning models.
    """

    def __init__(self):
        pass

    def get_model(self, X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2):
        """
        Trains a regression model using the provided training data.

        Args:
            X_train (np.array): Feature matrix for training.
            Y_train (np.array or pd.Series): Target variable for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of training data for validation.

        Returns:
            keras.Model: Trained Keras model.
        """
        # Ensure the input data are NumPy arrays
        #if isinstance(X_train, pd.DataFrame):
            #X_train = X_train.values
        if isinstance(Y_train, (pd.DataFrame, pd.Series)):
            Y_train = Y_train.values

        # Build a simple regression model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)  # Output layer for regression
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])


        # Fit the model with validation_split and EarlyStopping
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, )
        

        return model

    def save(self, model, filepath):
        """
        Saves the trained model to the specified filepath.

        Args:
            model (keras.Model): The trained Keras model.
            filepath (str): Path to save the model.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the model to the given path
        model.save(filepath)

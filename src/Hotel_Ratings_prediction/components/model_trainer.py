import os
import sys
import pandas as pd
import joblib
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.entity.config_entity import ModelTrainerConfig
from src.Hotel_Ratings_prediction.utils.common import load_df

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.X_train_transformed_df = config.X_train_transformed_df
        self.Y_train_transformed_df = config.Y_train_transformed_df
        self.X_test_transformed_df = config.X_test_transformed_df
        self.Y_test_transformed_df = config.Y_test_transformed_df
        self.model = config.model

    def build_model(self):
        try:
            X_train = load_df(self.X_train_transformed_df)
            X_test = load_df(self.X_test_transformed_df)
            Y_train = load_df(self.Y_train_transformed_df).values
            Y_test = load_df(self.Y_test_transformed_df).values

            if len(Y_train.shape) == 1:
                Y_train = Y_train.reshape(-1, 1)
            if len(Y_test.shape) == 1:
                Y_test = Y_test.reshape(-1, 1)

            logging.info("Starting model training with TensorFlow.")

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
            logging.info("Model trained successfully.")

            test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
            logging.info(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

            y_pred = model.predict(X_test).ravel()  # Ensure y_pred is 1D
            mse = mean_squared_error(Y_test, y_pred)
            mae = mean_absolute_error(Y_test, y_pred)
            logging.info(f"Test MSE: {mse}, Test MAE: {mae}")

            model.save(self.model)
            logging.info(f"Model saved to {self.model}")

            # Ensure the directory exists before saving plots
            plot_dir = 'artifacts/model_trainer/plots'
            os.makedirs(plot_dir, exist_ok=True)

            # Plot training history
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(history.history['loss'], label='Training Loss')
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0].set_title('Loss Over Epochs')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss (MSE)')
            axes[0].legend()

            axes[1].plot(history.history['mae'], label='Training MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('MAE Over Epochs')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('MAE')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'training_history.png'))
            logging.info("Training history plots saved successfully.")


            # Calculate residuals
            residuals = Y_test.ravel() - y_pred
            residuals = np.round(residuals, decimals=1)  # Round residuals to 1 decimal place

            residuals_pct = (residuals / Y_test.ravel()) * 100

            # Create results DataFrame
            results_df = pd.DataFrame({
                'actual': Y_test.ravel(), 
                'predicted': y_pred, 
                'diff': residuals, 
                'diff_pct': residuals_pct
            })
                        # Histograms of Predicted and Actual Values
            plt.figure(figsize=(10, 6))
            sns.histplot(results_df['diff'], kde=True)
            plt.title('Distribution of Residuals')
            plt.xlabel('Difference between Actual and predicted')
            plt.ylabel('No of instances')
            plt.savefig(os.path.join(plot_dir, 'histogram_actual_predicted.png'))
            logging.info("Histograms of Predicted and Actual Values saved successfully.")


            # QQ Plot of Residuals
            plt.figure(figsize=(8, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title("QQ Plot of Residuals")
            plt.savefig(os.path.join(plot_dir, 'qq_plot_residuals.png'))
            logging.info("QQ Plot of Residuals saved successfully.")

            # Residual Distribution Plot
            plt.figure(figsize=(10, 6))
            sns.histplot(results_df['diff'], kde=True)
            plt.title('Distribution of Residuals')
            plt.xlabel('Difference between Actual and Predicted')
            plt.ylabel('No of Instances')
            plt.savefig(os.path.join(plot_dir, 'residual_plots.png'))
            logging.info("Residual plots saved successfully.")

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise Exception(f"Error during training: {str(e)}")

import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.X_train_transformed_df = config.X_train_transformed_df
        self.Y_train_transformed_df = config.Y_train_transformed_df
        self.X_test_transformed_df = config.X_test_transformed_df
        self.Y_test_transformed_df = config.Y_test_transformed_df
        self.model = config.model

    def build_model(self):
        try:
            # Load training and testing data
            X_train = pd.read_csv(self.X_train_transformed_df)
            X_test = pd.read_csv(self.X_test_transformed_df)
            Y_train = pd.read_csv(self.Y_train_transformed_df).values
            Y_test = pd.read_csv(self.Y_test_transformed_df).values

            logging.info("Starting model training with RandomizedSearchCV.")

            # Define model choices and their hyperparameters
            models = {
                'random_forest': RandomForestRegressor(random_state=42),
                #'gradient_boosting': GradientBoostingRegressor(random_state=42),
                #'linear_regression': LinearRegression(),
                #'decision_tree': DecisionTreeRegressor(random_state=42),
                #'xgboost': xgb.XGBRegressor(random_state=42)
            }

            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                
                
                
            }

            best_model = None
            best_model_name = None
            best_params = None
            best_score = float('inf')  # Start with a high score to find the minimum MSE

            # Loop over each model to tune and evaluate
            for model_name in models:
                model = models[model_name]
                param_dist = param_grids[model_name]

                logging.info(f"Training {model_name} model.")

                # Hyperparameter tuning using RandomizedSearchCV
                if param_dist:
                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                                       n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
                    random_search.fit(X_train, Y_train.ravel())  # Fit model using the scaled data
                    best_model_candidate = random_search.best_estimator_
                    best_params_candidate = random_search.best_params_
                    logging.info(f"Best parameters for {model_name}: {best_params_candidate}")
                else:
                    model.fit(X_train, Y_train.ravel())  # For LinearRegression, no hyperparameter tuning
                    best_model_candidate = model
                    best_params_candidate = None

                # Model Evaluation
                Y_pred = best_model_candidate.predict(X_test)

                # Calculate performance metrics
                mse = mean_squared_error(Y_test, Y_pred)
                mae = mean_absolute_error(Y_test, Y_pred)
                r2 = r2_score(Y_test, Y_pred)

                logging.info(f"{model_name} - Test MSE: {mse}, Test MAE: {mae}, Test R2: {r2}")

                # Check if this model is the best one so far
                if mse < best_score:
                    best_model = best_model_candidate
                    best_model_name = model_name
                    best_params = best_params_candidate
                    best_score = mse

                 # Ensure the directory exists
            model_path = os.path.join(self.model, "models")
            ## Save model in artifacts/model_trainer/models/{model_name}_model.joblib
            if not os.path.exists(model_path):
                os.makedirs(model_path,exist_ok=True)
                best_model_path = os.path.join(model_path, f"{best_model_name}_model.joblib")
                logging.info(f"{model_name} model saved to {best_model_path}")
                
            os.makedirs(model_path, exist_ok=True)  # Creates 'models' directory if not exists

            # Save the model
            model_filename = f"{model_name}_model.joblib"
            model_file_path = os.path.join(model_path, model_filename)
            joblib.dump(best_model_candidate, model_file_path)

            logging.info(f"{model_name} model saved to {model_file_path}")

            # Log best model
            logging.info(f"Best model: {best_model_name} with MSE: {best_score}")
            logging.info(f"Best parameters: {best_params}")

            # Final evaluation of best model
            Y_pred = best_model.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            logging.info(f"Final model ({best_model_name}) - Test MSE: {mse}, Test MAE: {mae}, Test R2: {r2}")

            # Plot performance (Residuals and Actual vs Predicted)
            self.plot_performance(Y_test, Y_pred)

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise Exception(f"Error during training: {str(e)}")

    def plot_performance(self, Y_test, Y_pred):
        """Plot performance metrics and residuals"""
        # Residuals Plot
        residuals = Y_test.ravel() - Y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.savefig('artifacts/model_trainer/residuals.png')
        logging.info("Residuals plot saved successfully.")

        # Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, Y_pred)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('artifacts/model_trainer/actual_vs_predicted.png')
        logging.info("Actual vs Predicted plot saved successfully.")

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.cluster import KMeans
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.utils.common import create_directories, load_df
from src.Hotel_Ratings_prediction.entity.config_entity import DataPreparationConfig
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.input_path = config.input_path
        self.output_path = config.output_path
        self.preprocessor = config.preprocessor
        self.target_column = config.target_column
        self.X_train_transformed_df = config.X_train_transformed_df
        self.Y_train_transformed_df = config.Y_train_transformed_df
        self.X_test_transformed_df = config.X_test_transformed_df
        self.Y_test_transformed_df = config.Y_test_transformed_df

    def data_cleaning(self, df):
        """Clean the dataset by handling missing values and renaming columns."""
        logging.info("Starting data cleaning.")
        
        # Check initial shape
        logging.debug(f"Initial shape of the data: {df.shape}")

        # Fill missing values for numeric columns only (impute with the mean)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Handle missing categorical data by filling with 'Unknown'
        categorical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')

        # Standardize column names (lowercase and remove special characters)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        logging.info(f"Missing data after imputation: {df.isnull().sum()}")
        # Log the value counts of the target column
        logging.info(f"The value counts in target_columns are: {df[self.target_column].value_counts()}")

        # Ensure that 'aggregate_rating' is numeric (convert non-numeric to NaN)
        #y = pd.to_numeric(y, errors='coerce')  # This converts non-numeric values to NaN

        # Replace 0 values in 'aggregate_rating' with NaN
        y_replaced = df[self.target_column].replace(0.0, np.nan)
        
        # Use KNN Imputer to fill missing values (NaNs) using nearest neighbors
        knn_imputer = KNNImputer(n_neighbors=5,)
        y_filled = knn_imputer.fit_transform(y_replaced.values.reshape(-1, 1))

        # Now, update the 'aggregate_rating' column with the filled values
        df[self.target_column] = y_filled
        logging.info(f"Missing data after imputation: {df[self.target_column].value_counts()}")

        
        logging.debug(f"Data after cleaning: {df.shape}")
        return df

    def adding_columns(self, df):
        """Perform feature engineering by adding and transforming columns."""
        logging.info("Starting feature engineering: adding columns.")
        
        # Handle cuisines column more efficiently
        df[['cuisine1', 'cuisine2', 'cuisine3', 'cuisine4']] = df['cuisines'].str.split(', ', expand=True).iloc[:, :4]
        
        # Drop unnecessary columns safely
        df.drop(['cuisines', 'restaurant_name', 'switch_to_order_menu', 'country_code', 'restaurant_id'], 
                axis=1, errors='ignore', inplace=True)

        # Handle address column
        df[['floor', 'mall_name', 'street_name', 'city']] = df['address'].str.split(', ', expand=True).iloc[:, :4]
        df.drop(['address', 'locality', 'locality_verbose'], axis=1, errors='ignore', inplace=True)

        # Categorize price levels efficiently
        df['price_level'] = pd.cut(df['average_cost_for_two'], bins=[0, 500, 1000, 2000], 
                                labels=['Low', 'Medium', 'High'], include_lowest=True)

        logging.info("Feature engineering completed successfully.")
        return df

    def feature_engineering(self, df):
        """Perform feature engineering to prepare data for model training."""
        logging.info("Starting feature engineering for model training.")

        logging.debug(f"Initial shape of the dataframe: {df.shape}")

        # Split features and target
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        

                # Handle missing values in numeric columns by replacing zeros with NaN and filling with the mean
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_columns:
            X[col] = X[col].replace(0, np.nan)  # Replace 0 with NaN
            X[col].fillna(X[col].mean(), inplace=True)  # Fill NaN with the column mean

        # Split the data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Select all features for model training
        selected_features = numeric_columns + list(X.select_dtypes(exclude=['float64', 'int64']).columns)
        logging.info(f"Selected features: {selected_features}")

        # Filter the data with selected features only
        X_train_transformed_df = X_train[selected_features]
        X_test_transformed_df = X_test[selected_features]

        logging.debug(f"Final selected features: {selected_features}")
        logging.debug(f"Columns in X_train before transformation: {X_train_transformed_df.columns}")
        logging.debug(f"Columns in X_test before transformation: {X_test_transformed_df.columns}")

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(steps=[ 
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_columns = [col for col in X.select_dtypes(exclude=['float64', 'int64']).columns if col in selected_features]
        categorical_transformer = Pipeline(steps=[ 
            ('encoder', ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[ 
                ('numeric', numeric_transformer, numeric_columns),
                ('categorical', categorical_transformer, categorical_columns)
            ]
        )

        # Apply transformations
        X_train_transformed = preprocessor.fit_transform(X_train_transformed_df)
        X_test_transformed = preprocessor.transform(X_test_transformed_df)

        logging.debug(f"Transformed X_train shape: {X_train_transformed.shape}")
        logging.debug(f"Transformed X_test shape: {X_test_transformed.shape}")

        # Save the preprocessor
        joblib.dump(preprocessor, self.preprocessor)
        logging.info(f"Preprocessor saved to {self.preprocessor}")

        # Convert transformed data back to DataFrame
        X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=selected_features)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=selected_features)

        logging.info("Feature engineering for model training completed.")
        return X_train_transformed_df, X_test_transformed_df, Y_train, Y_test

    def preprocessing(self):
        """Run the full preprocessing pipeline."""
        logging.info("Starting preprocessing pipeline.")

        # Load data
        df = load_df(file_path=self.input_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Clean the data
        df = self.data_cleaning(df)
        logging.info("Data cleaning completed.")

        # Add columns through feature engineering
        df = self.adding_columns(df)
        logging.info("Columns added successfully.")

        # Perform feature engineering
        X_train_transformed_df, X_test_transformed_df, Y_train, Y_test = self.feature_engineering(df)

        # Save transformed data
        X_train_transformed_df.to_csv(self.X_train_transformed_df, index=False)
        pd.DataFrame(Y_train).to_csv(self.Y_train_transformed_df, index=False)
        
        X_test_transformed_df.to_csv(self.X_test_transformed_df, index=False)
        pd.DataFrame(Y_test).to_csv(self.Y_test_transformed_df, index=False)

        logging.info(f"Shape of X_train_transformed_df: {X_train_transformed_df.shape}")
        logging.info(f"Shape of Y_train: {Y_train.shape}")
        logging.info(f"Shape of X_test_transformed_df: {X_test_transformed_df.shape}")
        logging.info(f"Shape of Y_test: {Y_test.shape}")

        logging.info("Preprocessing pipeline completed successfully.")

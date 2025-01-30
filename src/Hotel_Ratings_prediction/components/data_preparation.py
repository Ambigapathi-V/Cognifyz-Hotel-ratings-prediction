import os
import sys
import pandas as pd
import yaml
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.cluster import KMeans
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.utils.common import create_directories, load_df
from src.Hotel_Ratings_prediction.entity.config_entity import DataPreparationConfig

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

        # Drop rows with missing values
        df = df.dropna()
        logging.info(f"Data after dropping missing values: {df.shape}")

        # Check and log missing data
        missing_data = df.isnull().sum()
        logging.info(f"Missing data summary:\n{missing_data}")

        # Standardize column names
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        logging.info(f"Standardized column names: {list(df.columns)}")

        return df

    def adding_columns(self, df):
        """Perform feature engineering by adding and transforming columns."""
        logging.info("Starting feature engineering: adding columns.")

        # Handle cuisines column
        df['cuisines'] = df['cuisines'].fillna('').str.split(', ')
        df['cuisine1'] = df['cuisines'].apply(lambda x: x[0] if len(x) > 0 else None)
        df['cuisine2'] = df['cuisines'].apply(lambda x: x[1] if len(x) > 1 else None)
        df['cuisine3'] = df['cuisines'].apply(lambda x: x[2] if len(x) > 2 else None)
        df['cuisine4'] = df['cuisines'].apply(lambda x: x[3] if len(x) > 3 else None)
        df.drop(columns=['cuisines','restaurant_id,switch_to_order_menu','country_code'], inplace=True)

        # Handle address column
        df['address'] = df['address'].fillna('').str.split(', ')
        df['floor'] = df['address'].apply(lambda x: x[0] if len(x) > 0 else None)
        df['mall_name'] = df['address'].apply(lambda x: x[1] if len(x) > 1 else None)
        df['street_name'] = df['address'].apply(lambda x: x[2] if len(x) > 2 else None)
        df['city'] = df['address'].apply(lambda x: x[3] if len(x) > 3 else None)
        df.drop(columns=['address', 'locality', 'locality_verbose'], inplace=True)

        # Categorize price levels
        price_bins = [0, 500, 1000, 2000]
        price_labels = ['Low', 'Medium', 'High']
        df['price_level'] = pd.cut(df['average_cost_for_two'], bins=price_bins, labels=price_labels, include_lowest=True)

        # Perform clustering on geographical data
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['geo_cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])

        # Calculate additional statistics
        df['city_restaurant_density'] = df['city'].map(df['city'].value_counts())
        df['average_city_rating'] = df['city'].map(df.groupby('city')['aggregate_rating'].mean())
        df['mall_popularity'] = df['mall_name'].map(df['mall_name'].value_counts())

        cuisine_popularity = pd.concat([df['cuisine1'], df['cuisine2'], df['cuisine3'], df['cuisine4']]).value_counts()
        df['cuisine_popularity'] = df['cuisine1'].map(cuisine_popularity)

        def extract_floor(floor):
            floor_mapping = {
                "Ground Floor": 0,
                "First Floor": 1,
                "Second Floor": 2,
                "Third Floor": 3,
                "Fourth Floor": 4
            }
            if isinstance(floor, str):
                for key in floor_mapping:
                    if key in floor:
                        return floor_mapping[key]
            return 0

        df['floor'] = df['floor'].apply(extract_floor)
        df['floor_rank'] = df['floor'].map(df['floor'].value_counts().rank(ascending=False))

        logging.info("Feature engineering completed.")
        return df


    def feature_engineering(self, df):
        """Perform feature engineering to prepare data for model training."""
        logging.info("Starting feature engineering for model training.")

        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Identify column types
        numeric_columns = X.select_dtypes(include=['float', 'int']).columns.tolist()
        categorical_columns = X.select_dtypes(exclude=['float', 'int']).columns.tolist()

        logging.info(f"Numeric Columns: {numeric_columns}")
        logging.info(f"Categorical Columns: {categorical_columns}")

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_columns),
                ('categorical', categorical_transformer, categorical_columns)
            ]
        )

        # Save the preprocessor
        joblib.dump(preprocessor, self.preprocessor)
        logging.info(f"Preprocessor saved to {self.preprocessor}")

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Apply transformations
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        X_train_transformed_df = pd.DataFrame(X_train_transformed)
        X_test_transformed_df = pd.DataFrame(X_test_transformed)
        
        logging.info("Feature engineering for model training completed.")
        return X_train_transformed_df, X_test_transformed_df, Y_train, Y_test

    def preprocessing(self):
        """Run the full preprocessing pipeline."""
        logging.info("Starting preprocessing pipeline.")

        # Load the data
        df = load_df(file_path=self.input_path)
        logging.info("Data loaded successfully.")

        # Clean the data
        df = self.data_cleaning(df)
        logging.info("Data cleaning completed.")

        # Add columns
        df = self.adding_columns(df)
        logging.info("Columns added successfully.")

        # Perform feature engineering
        X_train_transformed_df, X_test_transformed_df, Y_train, Y_test = self.feature_engineering(df)
        

        # Save transformed data
        X_train_transformed_df.to_csv(self.X_train_transformed_df )
        pd.DataFrame(Y_train).to_csv(self.Y_train_transformed_df, index=False)\
        
        X_test_transformed_df.to_csv(self.X_test_transformed_df)
        pd.DataFrame(Y_test).to_csv(self.Y_test_transformed_df, index=False)
        
        ## Print the shape of the transformed dataset
        logging.info(f"Shape of X_train_transformed_df: {X_train_transformed_df.shape}")
        logging.info(f"Shape of Y_train: {Y_train.shape}")
        logging.info(f"Shape of X_test_transformed_df: {X_test_transformed_df.shape}")
        logging.info(f"Shape of Y_test: {Y_test.shape}")
        
        logging.info("Preprocessing pipeline completed successfully.")

import os
import sys
from src.Hotel_Ratings_prediction.logging import logging
from src.Hotel_Ratings_prediction.Exception import Exception
from src.Hotel_Ratings_prediction.utils.common import create_directories, load_df
from src.Hotel_Ratings_prediction.entity.config_entity import DataPreparationConfig
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.cluster import KMeans

import pandas as pd
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
    
    def data_cleaning(self,df):      
        df = df.dropna()
        num_rows, num_columns = df.shape
        logging.info(f"Number of rows: {num_rows}")
        logging.info(f"Number of columns: {num_columns}")
          
        # Checking for missing values in each column
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / num_rows) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Missing Percentage': missing_percentage
        })
        logging.info(missing_data)

          # Conver the coluumns name as lowercase
        df.columns = df.columns.str.lower() # Convert column names to lower case
        df.columns = df.columns.str.replace(' ', '_') # Replace spaces with underscores
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '') # Remove special characters
        logging.info( 'Created column %s' % df.columns ) 
        return df
        
    def adding_columns(self,df):
    # Replace NaN values with an empty list
        df['cuisines'] = df['cuisines'].fillna('')
        logging.info("Replacing NaN values with an empty list")
        # Splitting the 'cuisines' column into a list of cuisines
        df['cuisines'] = df['cuisines'].str.split(', ')
        logging.info("Splitting 'cuisines' column into multiple columns")

        # Creating four new columns for the cuisines
        df['cuisine1'] = df['cuisines'].apply(lambda x: x[0] if len(x) > 0 else None)
        df['cuisine2'] = df['cuisines'].apply(lambda x: x[1] if len(x) > 1 else None)
        df['cuisine3'] = df['cuisines'].apply(lambda x: x[2] if len(x) > 2 else None)
        df['cuisine4'] = df['cuisines'].apply(lambda x: x[3] if len(x) > 3 else None)
        logging.info("Created new columns for cuisines")

    # Dropping the original 'cuisines' column
        df = df.drop(columns=['cuisines'])
        
                # Replace NaN values with an empty list
        df['address'] = df['address'].fillna('')

        # Splitting the 'address' column into a list of address
        df['address'] = df['address'].str.split(', ')

        # Creating four new columns for the cuisines
        df['floor'] = df['address'].apply(lambda x: x[0] if len(x) > 0 else None)
        df['mall_name'] = df['address'].apply(lambda x: x[1] if len(x) > 1 else None)
        df['street_name'] = df['address'].apply(lambda x: x[2] if len(x) > 2 else None)
        df['city'] = df['address'].apply(lambda x: x[3] if len(x) > 3 else None)
        logging.info("Created new columns for address")

        # Dropping the original 'cuisines' column
        df = df.drop(columns=['address','locality','locality_verbose'])
        logging.info("Created new columns for cuisines")

        
        # Categorizing average cost into different price levels: Low, Medium, High, Very High
        price_bins = [0, 500, 1000, 2000]  # Price ranges
        price_labels = ['Low', 'Medium', 'High']  # Labels for price categories
        logging.info("Price category labels creating...")

        # Assigning each restaurant to a price level based on average cost
        df['price_level'] = pd.cut(df['average_cost_for_two'], bins=price_bins, labels=price_labels, include_lowest=True)  # Formula: cut(average_cost_for_two, price_bins)
        
        kmeans = KMeans(n_clusters=5, random_state=42)  # Clustering into 5 groups
        df['geo_cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])  # Formula: KMeans(longitude, latitude)

        # City restaurant density (restaurants per city)
        city_counts = df['city'].value_counts()  # Formula: count(restaurants per city)
        df['city_restaurant_density'] = df['city'].map(city_counts)  # Formula: map(city_counts)
        
        # Average rating per city
        avg_city_rating = df.groupby('city')['aggregate_rating'].mean()  # Formula: mean(aggregate_rating by city)
        df['average_city_rating'] = df['city'].map(avg_city_rating)  # Formula: map(avg_city_rating)

        # Mall popularity (restaurants per mall)
        mall_counts = df['mall_name'].value_counts()  # Formula: count(restaurants per mall)
        df['mall_popularity'] = df['mall_name'].map(mall_counts)  # Formula: map(mall_counts)


        # Calculating the popularity of each cuisine across all restaurants
        cuisine_popularity = pd.concat([df['cuisine1'], df['cuisine2'], df['cuisine3'], df['cuisine4']]).value_counts()  # Formula: count(cuisines across all rows)
        df['cuisine_popularity'] = df['cuisine1'].map(cuisine_popularity)  # Formula: map(cuisine_popularity to cuisine1)

        def extract_floor(floor):
            # Map common floor names to numbers
            floor_mapping = {
                "Ground Floor": 0,
                "First Floor": 1,
                "Second Floor": 2,
                "Third Floor": 3,
                "Fourth Floor": 4
            }
            # Check if the floor matches the mapping
            if isinstance(floor, str):
                for key in floor_mapping:
                    if key in floor:
                        return floor_mapping[key]
            return 0

        # Apply the function to the floor column
        df["floor"] = df["floor"].apply(extract_floor)

        # Floor rank based on the number of restaurants per floor (higher count means higher rank)
        floor_rank = df['floor'].value_counts().rank(ascending=False)  # Formula: rank(floor by count)
        df['floor_rank'] = df['floor'].map(floor_rank)  # Formula: map(floor_rank)
        logging.info('Feature Engineering Completed...')
        return df
        
    def feature_engineering(self,df):
        # Define the target column
        y = df[self.target_column]  # target column

        # Drop 'aggregate_rating' column from the features (X)
        X = df.drop(columns=[self.target_column])

        import pandas as pd

        # Assuming X is your feature set, and y is your target

        # Ensure no duplicate columns in the categorical list
        categorical_columns = X.select_dtypes(exclude=['float', 'int']).columns.tolist()

        # Ensure only numeric columns are included in the numeric_columns list
        numeric_columns = X.select_dtypes(include=['float', 'int']).columns.tolist()

        # Check that only numeric columns are in numeric_columns
        logging.info(f"Numeric Columns: {numeric_columns}")
        logging.info(f"Categorical Columns: {categorical_columns}")

        # Define the transformers for numeric and categorical columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numeric values with mean
            ('scaler', StandardScaler())  # Standardize numeric columns
        ])

        # Use `OrdinalEncoder` from CategoryEncoders to handle categorical columns with unseen categories
        categorical_transformer = Pipeline([
            ('encoder', ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute'))  # Impute unknown categories
        ])

        # Combine numeric and categorical transformers into one preprocessor pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_columns),  # Apply numeric transformations
                ('categorical', categorical_transformer, categorical_columns)  # Apply ordinal encoding to categorical columns
            ]
        )


        # Save the preprocessor to a file
        joblib.dump(preprocessor, self.preprocessor)
        logging.info("Preprocessor saved successfully! in %s" % self.preprocessor)

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Apply preprocessing
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Print shapes to check the transformed data
        logging.info("X_train_transformed shape:", X_train_transformed.shape)
        logging.info("X_test_transformed shape:", X_test_transformed.shape)
        # Convert the transformed data back to a Pandas DataFrame
        X_train_transformed_df = pd.DataFrame(X_train_transformed)

        # Check for null values
        print(X_train_transformed_df.isnull().sum().sum())
        print(Y_train.isnull().sum())

        # check for X_test_transformed_df
        X_test_transformed_df = pd.DataFrame(X_test_transformed)

        print(X_test_transformed_df.isnull().sum().sum())
        print(Y_test.isnull().sum())
        X_test_transformed_df.fillna(0,inplace=True)
        print(X_test_transformed_df.isnull().sum().sum())
        
        return X_train_transformed_df, X_test_transformed_df, Y_train, Y_test
        
    def preprocessing(self):
        ## Load the data
        df = load_df(file_path=self.input_path)
        
        ## Data cleaning
        df = self.data_cleaning(df)
        
        ## Adding columns
        df = self.adding_columns(df)
        
        ## Feature engineering
        X_train_transformed_df, X_test_transformed_df, Y_train, Y_test = self.feature_engineering(df)
        
        Y_train_transformed_df = pd.DataFrame(Y_train)
        Y_test_transformed_df = pd.DataFrame(Y_test)
                
        ## Save the X_train_transformed_df and Y_train to csv files
        X_train_transformed_df.to_csv(self.X_train_transformed_df, index=False)
        Y_train_transformed_df.to_csv(self.Y_train_transformed_df, index=False)
        X_test_transformed_df.to_csv(self.X_test_transformed_df, index=False)
        Y_test_transformed_df.to_csv(self.Y_test_transformed_df, index=False)
        
        logging.info("Data preprocessing completed!")
        
        
        
    
        




            
            
                
                

          
                

![vedio](img\project.jpg)

# Restaurant Ratings prediction(Cognifyz Technologies)

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction)
![GitHub](https://img.shields.io/github/license/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction)
![contributors](https://img.shields.io/github/contributors/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction) 
![codesize](https://img.shields.io/github/languages/code-size/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction) 

 

## Project Overview

The Cognifyz Technologies Data Science Internship Program is designed to provide hands-on experience in data science, machine learning, and artificial intelligence. Interns will work on real-world problems, gaining exposure to data exploration, preprocessing, modeling, and evaluation. The program focuses on providing tools and knowledge to empower individuals and organizations to harness the power of data.
## Features

- Data Exploration and Preprocessing
- Predictive Modeling
- Customer Analysis
- Geospatial Analysis
- Data Visualization
- Feature Engineering


## Demo

[App Link](https://hotel-ratings-predictor.streamlit.app/)


## Screenshots

![App Screenshot](img\hotel_page-0001.jpg)

# Installation and Setup

## **Installation and Setup**

In this section, we’ll walk through the process of setting up the **Cognifyz-Hotel-Ratings-Prediction** project on your local machine. Follow the steps below to install the necessary dependencies and get the project up and running.


## **Codes and Resources Used**

In this section, we provide the necessary information about the software and tools used to develop and run this project.

- **Editor Used**:  
  - **VS Code** (Visual Studio Code) was used as the primary editor for writing and developing the code. You can download it from [here](https://code.visualstudio.com/).
  - **Jupyter Notebook** was used for running interactive Python code, performing data analysis, and visualizing results. It’s part of the **Anaconda** distribution, which can be downloaded [here](https://www.anaconda.com/products/individual).

- **Python Version**:  
  This project was developed using **Python 3.x**. You can check your Python version by running:
  ```bash
  python --version
 ``

## Python packages Used

The project uses the following Python libraries:
- **Pandas** for data manipulation and analysis.
- **Numpy** for numerical operations.
- **Scikit-learn** for machine learning algorithms and model - evaluation.
- **Matplotlib** and Seaborn for data visualization.
- **Geopy** for geospatial analysis (if applicable). You can install all required libraries via:
```bash
pip install -r requirements.txt
```

## Data

The very crucial part of any data science project is dataset. Therefore list all the data sources used in the project, including links to the original data, descriptions of the data, and any pre-processing steps that were taken.

I structure this as follows - 

## **Data**

The very crucial part of any data science project is the dataset. In this section, we outline the data sources used in the **Restaurant Ratings Prediction** project, provide descriptions of the data, and explain any pre-processing steps taken to prepare it for analysis.

### **Source Data**

- **Restaurant Ratings Dataset**:  
  - **Source**: Provided by **Cognifyz Technologies**  
  - **Description**: The dataset contains information about various restaurants, including features such as restaurant name, location, price range, amenities, and customer ratings. The target variable is the restaurant rating, which is the value we aim to predict based on the input features such as price, location, and amenities.

- **Additional Data (if applicable)**:  
  - **Source**: Provided by **Cognifyz Technologies**  
  - **Description**: Any additional datasets such as information on restaurant types, cuisine types, or historical ratings that may be relevant to improving the predictive accuracy of the model.

### **Data Ingestion**

The dataset was provided directly by **Cognifyz Technologies**. It was delivered in CSV format and stored locally for further processing. This file includes all the necessary data for the project, which was loaded into Python using **Pandas** for analysis.

### **Data Preprocessing**

Once the data was acquired, several preprocessing steps were carried out to clean and prepare it for model training and evaluation:

- **Handling Missing Values**:  
  - Missing values in numerical columns (e.g., `price`, `rating`) were handled by replacing them with the median value for that feature.  
  - For categorical columns (e.g., `location`, `amenities`), missing values were imputed with the mode or categorized as "Unknown".

- **Data Normalization/Standardization**:  
  - **Price** and other continuous variables were normalized using **Min-Max scaling**, bringing all values within the range of 0 to 1.  
  - Other numerical features were standardized using **Z-score normalization** (subtracting the mean and dividing by the standard deviation) to ensure all features have similar scales.

- **Feature Engineering**:  
  - **Price per Rating**: A new feature was derived by dividing the `price` by the `rating` to understand how price correlates with ratings.  
  - **Location Encoding**: The categorical variable `location` was one-hot encoded to make it usable for machine learning models.  
  - **Amenities Count**: A feature was created to count the number of amenities offered by each restaurant, which could influence the restaurant rating.

- **Outlier Detection**:  
  - Outliers in numerical features like `price` were detected using the **IQR (Interquartile Range)** method and removed to prevent skewed results from outlier values.

- **Data Splitting**:  
  - The dataset was split into a **training set** (80%) for model training and a **test set** (20%) for model evaluation. This split was performed using **Stratified K-Folds Cross Validation** to ensure a consistent distribution of ratings across both sets.

### **Preprocessing Code**

The preprocessing steps were implemented in the `data_preprocessing.py` script or the corresponding Jupyter notebook (`trails.ipynb`). You can check these files for the detailed code used to process the data.

## **Code Structure**

The project is organized in a clear and modular structure, making it easy to navigate. Below is an overview of the project structure and the purpose of each file:

```bash
├── data     
  # Folder containing raw and cleaned datasets
├── src/Hotel_Ratings_prediction 
# Code for predicting hotel ratings using the model
├── .gitignore                  
# Specifies which files to exclude from Git version control
├── README.md                   
# Documentation of the project
├── api.py                      
# API script for making predictions
├── app.py                      
# Main application script for running the model
├── demo.py                     
# Demo script to show functionality of the model
├── requirements.txt            
# List of required Python dependencies
├── setup.py                    
# Setup script for configuring the environment
└── LICENSE                     
```

## **Result and Evaluation**

The model was trained using **Random Forest Regressor**, yielding the following performance metrics:

- **Mean Squared Error (MSE)**: 0.0284  
- **Mean Absolute Error (MAE)**: 0.1133  
- **R-squared (R²)**: 0.8781  

These results indicate that the model performs well, with a high R² score showing it explains 87.8% of the variance in restaurant ratings. The low MSE and MAE suggest accurate predictions. The **Actual vs Predicted** and **Residuals plots** provide further insight into the model's performance.

![model](artifacts\model_trainer\actual_vs_predicted.png)
![model](artifacts\model_trainer\residuals.png)


## **Future Work**

There are several potential avenues to extend and improve the project:

1. **Incorporate More Data**: Integrating additional features like customer reviews, restaurant popularity, and location data could enhance model performance.
2. **Model Experimentation**: Experiment with different machine learning algorithms (e.g., Gradient Boosting, Neural Networks) to compare results.
3. **Real-time Predictions**: Develop a live prediction API for real-time restaurant rating predictions.
4. **User Interface**: Create a web or mobile app interface for easier user interaction.
5. **Expand Dataset**: Use global datasets for a more generalized model.


## **Deployment**

To deploy this project, follow these steps:

1. Ensure all dependencies are installed.
2. Run the following command to deploy the project:
   ```bash
   python app.py
    ```

## Installation

To install the project locally, follow these instructions:
1. Clone the repository:
```bash
git clone https://github.com/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction.git
```
2. Navigate to the project directory:
```bash
cd Restaurant-Ratings-Prediction
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
python app.py
```

    
## Acknowledgements

## **Acknowledgements**

We would like to acknowledge the following contributors and resources that have played a significant role in the success of this project:

- **Cognifyz Technologies** for providing the dataset.
- **Scikit-learn**, **Pandas**, **NumPy**, and other libraries for enabling data preprocessing and machine learning model development.
- **GitHub** for hosting the project and enabling collaborative development.
- **Matplotlib** and **Seaborn** for data visualization.

Thank you to all contributors for their support and guidance.

## **License**

This project is licensed under the **MIT License**. You can freely use, modify, and distribute this code, provided you include a copy of the license.

- **MIT License**: 

The dataset used in this project was provided by **Cognifyz Technologies**. Please ensure you comply with any terms or conditions set by the dataset provider when using or sharing the data.

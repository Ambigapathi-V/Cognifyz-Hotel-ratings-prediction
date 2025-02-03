from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import requests

app = Flask(__name__)

# Load preprocessor and model
preprocessor = load('artifacts/data_preparation/preprocessor.joblib')
model = load(r'artifacts\model_trainer\model.joblib\models\random_forest_model.joblib')

# Function to get latitude and longitude from city using Maps.co API
def get_lat_lon_from_mapsco(location, api_key):
    """Get latitude and longitude from a location using Maps.co API."""
    url = f"https://geocode.maps.co/search?q={location}&api_key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        if result:
            latitude = result[0]['lat']
            longitude = result[0]['lon']
            return latitude, longitude
    return None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_rating = None
    
    if request.method == 'POST':
        # Get form data
        city = request.form.get('city')
        cuisine1 = request.form.get('cuisine1')
        cuisine2 = request.form.get('cuisine2')
        price_range = request.form.get('price_range')
        rating_color = request.form.get('rating_color')
        street_name = request.form.get('street_name')
        price_level = request.form.get('price_level')
        rating_text = request.form.get('rating_text')
        is_delivering_now = request.form.get('is_delivering_now')
        average_cost_for_two = request.form.get('average_cost_for_two')
        votes = request.form.get('votes')
        has_online_delivery = request.form.get('has_online_delivery')
        has_table_booking = request.form.get('has_table_booking')
        floor = request.form.get('floor')
        mall_name = request.form.get('mall_name')

        # Fetch latitude and longitude
        api_key = "your_maps_api_key"
        latitude, longitude = get_lat_lon_from_mapsco(city, api_key)
        
        # Prepare DataFrame with user inputs
        df = pd.DataFrame({
            'city': [city],
            'average_cost_for_two': [average_cost_for_two],
            'price_range': [price_range],
            'votes': [votes],
            'currency': ['Dollar($)'],  # Placeholder for currency
            'has_table_booking': [has_table_booking],
            'is_delivering_now': [is_delivering_now],
            'rating_color': [rating_color],
            'rating_text': [rating_text],
            'cuisine1': [cuisine1],
            'cuisine2': [cuisine2],
            'mall_name': [mall_name],
            'street_name': [street_name],
            'has_online_delivery': [has_online_delivery],
            'floor': [floor],
            'price_level': [price_level],
            'latitude': [latitude],
            'longitude': [longitude],
        })
        
        # Preprocess the data
        input_data_processed = preprocessor.transform(df)

        # Make prediction
        predicted_rating = model.predict(input_data_processed)[0]

    return render_template('index.html', predicted_rating=predicted_rating)

if __name__ == '__main__':
    app.run(debug=True)

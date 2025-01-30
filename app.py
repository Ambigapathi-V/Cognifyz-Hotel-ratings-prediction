import streamlit as st
import pandas as pd
import requests
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load preprocessor and model
preprocessor = load('artifacts/data_preparation/preprocessor.joblib')
model = load_model('artifacts/model_trainer/model.h5', custom_objects={'mse': MeanSquaredError()})

# Streamlit UI
st.title("🍽️ Restaurant Aggregate Rating Prediction")

# Country Mapping for user input
country_mapping = {
    "India": 1,
    "United States": 2,
    "United Kingdom": 3
}

# Cuisine options
cuisine_options = [
    'French', 'Japanese', 'Chinese', 'Filipino', 'American', 'Korean', 'Cafe',
    'Italian', 'Fast Food', 'Bakery', 'Brazilian', 'Pizza', 'Arabian', 'Bar Food',
    'Mexican', 'International', 'Peruvian', 'Seafood', 'Desserts', 'Juices',
    'Beverages', 'Lebanese', 'Burger', 'Steak', 'Indian', 'Sushi', 'BBQ',
    'Gourmet Fast Food', 'Coffee and Tea', 'Asian', 'Southern', 'Breakfast',
    'Sandwich', 'German', 'Vietnamese', 'Thai', 'Modern Australian', 'European',
    'Latin American', 'Mediterranean', 'Tea', 'Greek', 'Spanish', 'Hawaiian',
    'Irish', 'New American', 'Caribbean', 'Cajun', 'Pub Food', 'Tapas',
    'Singaporean', 'Western', 'Finger Food', 'British', 'Cuban', 'Australian',
    'Turkish', 'Pakistani', 'Continental', 'Goan', 'South Indian', 'African',
    'North Indian', 'Rajasthani', 'Ice Cream', 'Mughlai', 'Street Food', 'Mithai',
    'Maharashtrian', 'Biryani', 'Parsi', 'Raw Meats', 'Kashmiri', 'Persian',
    'Healthy Food', 'Portuguese', 'Burmese', 'Awadhi', 'Bengali', 'Tibetan',
    'Lucknowi', 'Gujarati', 'Oriya', 'Hyderabadi', 'Bihari', 'Afghani', 'Kerala',
    'Andhra', 'Malwani', 'Cuisine Varies', 'Assamese', 'North Eastern', 'Nepalese',
    'Naga', 'Modern Indian', 'Salad', 'Middle Eastern', 'Charcoal Grill',
    'Asian Fusion', 'Taiwanese', 'Kiwi', 'Malaysian', 'Contemporary', 'Scottish',
    'Ramen', 'Argentine', 'Grill', 'Kebab', 'Patisserie', 'World Cuisine',
    'Turkish Pizza', 'Restaurant Cafe', 'None'
]

# Sidebar Input Fields
restaurant_name = st.text_input("Restaurant Name")
country_name = st.selectbox("Country", list(country_mapping.keys()))
city = st.text_input("City")
average_cost_for_two = st.number_input("Average Cost for Two people")
price_range = st.selectbox("Price Range", [1, 2, 3, 4, 5])
votes = st.number_input("Votes")

currency = st.selectbox("Currency",['Botswana Pula(P)', 'Brazilian Real(R$)', 'Dollar($)',
       'Emirati Dirham(AED)', 'Indian Rupees(Rs.)', 'New Zealand($)',
       'Pounds(£)', 'Qatari Rial(QR)', 'Rand(R)',
       'Sri Lankan Rupee(LKR)', 'Turkish Lira(TL)'])
has_table_booking = st.selectbox("Has Table Booking",['Yes', 'No'])
is_delivering_now = st.selectbox("Is Delivering Now",['Yes', 'No'])
rating_color = st.selectbox("Rating Color",['Dark Green', 'Green', 'Yellow', 'Orange', 'Red'])
rating_text = st.selectbox("Rating Text",['Excellent', 'Good', 'Average', 'Not Good', 'Poor'])
cuisines = st.multiselect("Cuisines", cuisine_options)

# Prepare DataFrame
df = pd.DataFrame({
    'restaurant_name': [restaurant_name],
    'city': [city],
    'average_cost_for_two': [average_cost_for_two],
    'price_range': [price_range],
    'votes': [votes],
    'currency': [currency],
    'has_table_booking': [has_table_booking],
    'is_delivering_now': [is_delivering_now],
    'rating_color': [rating_color],
    'rating_text': [rating_text],
    'cuisines': [cuisines]
})

# Categorizing average cost into different price levels
price_bins = [0, 500, 1000, 2000]  # Price ranges
price_labels = ['Low', 'Medium', 'High']  # Labels for price categories
df['price_level'] = pd.cut(df['average_cost_for_two'], bins=price_bins, labels=price_labels, include_lowest=True) 

# Replace NaN values with an empty list (for cuisines column)
df['cuisines'] = df['cuisines'].fillna('')

# If cuisines column is not a list, convert it to a list
df['cuisines'] = df['cuisines'].apply(lambda x: x if isinstance(x, list) else [])

# Creating four new columns for the cuisines
df['cuisine1'] = df['cuisines'].apply(lambda x: x[0] if len(x) > 0 else None)
df['cuisine2'] = df['cuisines'].apply(lambda x: x[1] if len(x) > 1 else None)
df['cuisine3'] = df['cuisines'].apply(lambda x: x[2] if len(x) > 2 else None)
df['cuisine4'] = df['cuisines'].apply(lambda x: x[3] if len(x) > 3 else None)

# Dropping the original 'cuisines' column
df = df.drop(columns=['cuisines'])

# Counting the number of cuisines for each restaurant
df['cuisine_count'] = df[['cuisine1', 'cuisine2', 'cuisine3', 'cuisine4']].notna().sum(axis=1)

# Convert selected country to corresponding code
country_code = country_mapping[country_name]

# Geocoding: Get latitude and longitude using PositionStack API
def get_coordinates_from_positionstack(location, api_key):
    geocode_url = f"http://api.positionstack.com/v1/forward?access_key={api_key}&query={location}"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        result = response.json()
        if result['data']:
            latitude = result['data'][0]['latitude']
            longitude = result['data'][0]['longitude']
            return latitude, longitude
    return None, None

# Use the PositionStack API to get latitude and longitude
api_key = 'fad6ef4590854a9bde4e010bd94ffeb9'  # Replace with your actual API key
location = city  # City entered by user in Streamlit input
latitude, longitude = get_coordinates_from_positionstack(location, api_key)

# Add latitude and longitude to the DataFrame
df['latitude'] = latitude
df['longitude'] = longitude

if latitude and longitude:
    st.success(f"Latitude: {latitude}, Longitude: {longitude}")
else:
    st.warning("Location not found.")

# Prediction and display of results
df['country_code'] = country_code
input_data_processed = preprocessor.transform(df)

prediction = model.predict(input_data_processed)
predicted_rating = prediction[0][0]

st.subheader(f"Predicted Restaurant Rating: ⭐ {predicted_rating:.2f}")

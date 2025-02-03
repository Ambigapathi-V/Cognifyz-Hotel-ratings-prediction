import streamlit as st
import pandas as pd
from joblib import load
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load preprocessor and model
preprocessor = load('artifacts/data_preparation/preprocessor.joblib')
model = load(r'artifacts/model_trainer/model.joblib/models/random_forest_model.joblib')

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

# UI Title
st.markdown("<h1 style='text-align: center;'>🍽️ Restaurant Rating Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict restaurant ratings based on key features</h3>", unsafe_allow_html=True)
st.write("---")

# Layout with 3 columns for proper alignment
col1, col2, col3 = st.columns(3)
cities_options = [
    "New York", "London", "Delhi", "Paris", "Tokyo",
    "New Delhi", "Noida", "Gurgaon", "Faridabad", "Near",
    "Karol Bagh", "Shahdara", "Mahipalpur", "Pitampura",
    "Mayur Vihar Phase 1", "Rajinder Nagar", "Safdarjung",
    "Kirti Nagar", "Delhi University-GTB Nagar", "Saket",
    "Laxmi Nagar", "Tilak Nagar", "Connaught Place",
    "Kamla Nagar", "Malviya Nagar", "Palam", "Kalkaji",
    "Rajouri Garden", "MG Road", "Tagore Garden"
]

cusine_varities =[
    "North Indian", "Chinese", "Fast Food", "Bakery", "Cafe", 
    "South Indian", "American", "Mithai", "Street Food", "Pizza",
    "Mughlai", "Ice Cream", "Italian", "Desserts", "Continental",
    "Burger", "Biryani", "Raw Meats", "Beverages", "Healthy Food",
    "Asian", "Indian", "Mexican", "Japanese", "Seafood", "Breakfast",
    "European", "Brazilian", "Finger Food", "Tibetan", "Lebanese",
    "Thai", "BBQ", "Tea", "Bengali", "Kerala", "Mediterranean",
    "Kashmiri", "International", "Juices", "Steak", "Sushi", "British",
    "French", "Kebab", "Coffee and Tea", "Hyderabadi", "Bar Food",
    "Goan", "Pakistani", "Afghani", "Latin American", "Gujarati",
    "Filipino", "Maharashtrian", "Turkish", "Lucknowi", "Naga",
    "Greek", "Awadhi", "Southern", "Sandwich", "Arabian", "Hawaiian",
    "Rajasthani", "Parsi", "Bihari", "Restaurant Cafe", "Contemporary",
    "North Eastern", "Portuguese", "Nepalese", "German", "Caribbean",
    "Korean"
]
currency_types = [
    "Indian Rupees(Rs.)", "Dollar($)", "Pounds(£)", "Emirati Diram(AED)",
    "Brazilian Real(R$)", "Rand(R)", "NewZealand($)", "Turkish Lira(TL)",
    "Qatari Rial(QR)", "Botswana Pula(P)", "Sri Lankan Rupee(LKR)"
]



# Left column
with col1:
    city = st.selectbox("🏙️ City", cities_options)
    cuisine1 = st.selectbox("🍴 Cuisine 1", cusine_varities)
    cuisine2 = st.selectbox("🍴 Cuisine 2", cusine_varities)
    price_range = int(st.selectbox("💲 Price Range", [1,2,3,4,5]))
    rating_color = st.selectbox("🎨 Rating Color", ["Green", "Yellow", "Orange", "Red", "Dark Green"])
    street_name = st.text_input("🛣️ Street Name")

# Middle column
with col2:
    cuisine3 = st.selectbox("🍴 Cuisine 3", cusine_varities)
    cuisine4 = st.selectbox("🍴 Cuisine 4", cusine_varities)
    price_level = st.selectbox("💰 Price Level", ["Low", "Medium", "High"])
    currency = st.selectbox("💵 Currency", currency_types)
    rating_text = st.selectbox("📊 Rating Text", ["Excellent", "Good", "Average", "Not Good", "Poor"])
    is_delivering_now = st.selectbox("🚚 Is Delivering Now?", ["Yes", "No"])

# Right column
with col3:
    average_cost_for_two = int(st.number_input("💰 Average Cost for Two People", min_value=1))
    votes = int(st.number_input("👍 Votes", min_value=0))
    has_online_delivery = st.selectbox("📦 Has Online Delivery?", ["Yes", "No"])
    has_table_booking = st.selectbox("🍽️ Has Table Booking?", ["Yes", "No"])
    floor = int(st.selectbox("🏬 Floor", [1, 2, 3, 4, 5]))
    mall_name = st.text_input("🏢 Mall Name")


# Fetch latitude and longitude using the Maps.co API when city is selected
api_key = "679c8a06120d8228223481dyna5e88b"
latitude, longitude = get_lat_lon_from_mapsco(city, api_key)

if latitude and longitude:
    st.write(f"📍 Latitude: {latitude}, Longitude: {longitude}")
else:
    st.write("❌ Location not found. Please try again with a valid location.")

# Prepare DataFrame with user inputs and the fetched latitude/longitude
df = pd.DataFrame({
    'city': [city],
    'average_cost_for_two': [average_cost_for_two],
    'price_range': [price_range],
    'votes': [votes],
    'currency': [currency],
    'has_table_booking': [has_table_booking],
    'is_delivering_now': [is_delivering_now],
    'rating_color': [rating_color],
    'rating_text': [rating_text],
    'cuisine1': [cuisine1],
    'cuisine2': [cuisine2],
    'cuisine3': [cuisine3],
    'cuisine4': [cuisine4],
    'mall_name': [mall_name],
    'street_name': [street_name],
    'has_online_delivery': [has_online_delivery],
    'floor': [floor],
    'price_level': [price_level],
    'latitude': [latitude],  # Adding latitude
    'longitude': [longitude]  # Adding longitude
})

# Preprocess the DataFrame
input_data_processed = preprocessor.transform(df)


predicted_rating = model.predict(input_data_processed)[0]

# Prediction Button
st.write(" ")
if st.button("🔮 Predict Rating"):
    if predicted_rating is not None:
        st.success(f"Predicted Restaurant Rating: ⭐{predicted_rating:.2f}")
    else:
        st.error("Prediction failed due to input issues.")

# Footer
st.markdown('<p class="footer">🔗 Developed by Ambigapathi V | Powered by Machine Learning</p>', unsafe_allow_html=True)

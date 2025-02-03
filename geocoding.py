import requests

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

# Replace with your actual API key
api_key = "679c8a06120d8228223481dyna5e88b"

location = input("Enter location: ")
latitude, longitude = get_lat_lon_from_mapsco(location, api_key)

if latitude and longitude:
    print(f"📍 Location: {location}\n🌍 Latitude: {latitude}, Longitude: {longitude}")
else:
    print("❌ Location not found. Please try again with a valid location.")

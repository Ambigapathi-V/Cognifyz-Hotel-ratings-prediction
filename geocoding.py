import requests

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

# Replace 'YOUR_API_KEY' with your actual PositionStack API key
api_key = 'fad6ef4590854a9bde4e010bd94ffeb9'
location = "salem"
latitude, longitude = get_coordinates_from_positionstack(location, api_key)

if latitude and longitude:
    print(f"Latitude: {latitude}, Longitude: {longitude}")
else:
    print("Location not found.")

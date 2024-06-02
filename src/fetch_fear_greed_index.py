import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch Fear and Greed Index
def fetch_fear_greed_index(api_key):
    url = "https://api.alternative.me/fng/"
    params = {
        'limit': 0,  # Fetch all available data
        'format': 'json',  # Get data in JSON format
        'date_format': 'world'  # Get dates in DD/MM/YYYY format
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        index_data = response.json().get('data', [])
        return index_data
    else:
        print("Failed to fetch data: HTTP Status Code", response.status_code)
        return []

# Define the API key (no need to change)
API_KEY = 'your_api_key_here'

# Fetch the data
index_data = fetch_fear_greed_index(API_KEY)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(index_data)
df.to_csv('fear_and_greed_index.csv', index=False)

print("Data fetched and saved successfully. Please check the 'fear_and_greed_index.csv' file.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved model
model = joblib.load('src/models/bitcoin_prediction_model.pkl')

# Load the saved scaler
scaler = joblib.load('src/models/scaler.pkl')

# Example new data
new_data = {
    'Open': [45000],
    'Price': [45500],
    'High': [46000],
    'Low': [44000],
    'SN&P Adjusted': [5000],
    'GOLD Adjusted': [1800],
    'Days from the last halving': [100],
    'BTC_Hashprice': [0.000002],
    'Crypto Volatility Index': [70],
    'support_level': [44000],
    'resistance_level': [46000],
    'ETH Price': [3000],
    'ETH Vol.': [1e6],
    'OIL Price Adjusted': [70]
}

new_data_df = pd.DataFrame(new_data)

# Define features
features = [
    'Open', 'Price', 'High', 'Low', 
    'SN&P Adjusted', 'GOLD Adjusted', 
    'Days from the last halving', 'BTC_Hashprice', 
    'Crypto Volatility Index', 'support_level', 
    'resistance_level', 'ETH Price', 'ETH Vol.', 
    'OIL Price Adjusted'
]

# Fill NaN values if any
new_data_df.fillna(new_data_df.mean(), inplace=True)

# Scale the features
new_data_scaled = scaler.transform(new_data_df[features])

# Make predictions
predictions = model.predict(new_data_scaled)

# Print predictions
print("Predictions:", predictions)

# Print predictions with a descriptive message
if predictions[0] == 1:
    print("Prediction: The change in Bitcoin price will be above 3%.")
else:
    print("Prediction: The change in Bitcoin price will not be above 3%.")
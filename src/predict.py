import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset from Excel file
file_path = 'data/Bitcoin-Historical-Data.xlsx'  # Use relative path if possible
data = pd.read_excel(file_path)

# Function to convert 'K', 'M', and 'B' values to numeric
def convert_kmb_to_numeric(col):
    col = col.replace({'K': '*1e3', 'M': '*1e6', 'B': '*1e9'}, regex=True).map(pd.eval).astype(float)
    return col

# Ensure these columns exist in the dataset
if 'Vol.' in data.columns and 'ETH Vol.' in data.columns:
    data['Vol.'] = convert_kmb_to_numeric(data['Vol.'])
    data['ETH Vol.'] = convert_kmb_to_numeric(data['ETH Vol.'])
else:
    raise KeyError("Expected columns 'Vol.' or 'ETH Vol.' not found in the dataset.")

# Placeholder example for support and resistance levels calculation
def calculate_support_resistance(data):
    data['support_level'] = data['Low'].rolling(window=10).min()
    data['resistance_level'] = data['High'].rolling(window=10).max()

calculate_support_resistance(data)

# Ensure support_level and resistance_level columns exist and fill NaN values
data['support_level'].fillna(0, inplace=True)
data['resistance_level'].fillna(0, inplace=True)

# Preprocessing
# Fill missing values if any (you can use more sophisticated methods if needed)
data.ffill(inplace=True)
data.bfill(inplace=True)

# Define features and target
features = [
    'Open', 'Price', 'High', 'Low',
    'SN&P Adjusted', 'GOLD Adjusted',
    'Days from the last halving', 'BTC_Hashprice',
    'Crypto Volatility Index', 'support_level',
    'resistance_level', 'ETH Price', 'ETH Vol.',
    'OIL Price Adjusted'
]

# Check if all features exist in the dataset
missing_features = [feat for feat in features if feat not in data.columns]
if missing_features:
    raise KeyError(f"Missing features in the dataset: {missing_features}")

X = data[features]
y = data.get('Target Value')

if y is None:
    raise KeyError("Target column 'Target Value' not found in the dataset.")

# Check for any remaining NaNs
print("Number of NaNs in each feature before handling:")
print(X.isna().sum())

# Fill remaining NaNs with mean or median
X.fillna(X.mean(), inplace=True)

# Verify no NaNs remain
print("Number of NaNs in each feature after handling:")
print(X.isna().sum())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'src/models/scaler.pkl')  # Use relative path if possible

# Hyperparameter tuning using GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_scaled, y_train)

# Use the best Random Forest estimator
best_rf_model = grid_search_rf.best_estimator_

# Make predictions with Random Forest
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
print("Random Forest Model Evaluation:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print('Accuracy:', accuracy_score(y_test, y_pred_rf))

# Save the Random Forest model
joblib.dump(best_rf_model, 'src/models/bitcoin_prediction_rf_model.pkl')  # Use relative path if possible

# Hyperparameter tuning using GridSearchCV for Logistic Regression
param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l2', 'none']
}

grid_search_lr = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=1000), param_grid=param_grid_lr, cv=3, n_jobs=-1, verbose=2)
grid_search_lr.fit(X_train_scaled, y_train)

# Use the best Logistic Regression estimator
best_lr_model = grid_search_lr.best_estimator_

# Make predictions with Logistic Regression
y_pred_lr = best_lr_model.predict(X_test_scaled)

# Evaluate the Logistic Regression model
print("Logistic Regression Model Evaluation:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print('Accuracy:', accuracy_score(y_test, y_pred_lr))

# Save the Logistic Regression model
joblib.dump(best_lr_model, 'src/models/bitcoin_prediction_lr_model.pkl')  # Use relative path if possible

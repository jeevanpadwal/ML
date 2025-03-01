# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Step 1: Load and filter the datasets
market_data = pd.read_csv('D:/programming/Python/ML/4/onion_pune.csv')
weather_data = pd.read_csv('D:/programming/Python/ML/4/pune_with_date.csv')

# Filter market data for onions in Pune
market_data = market_data[(market_data['Market'] == 'Pune') & (market_data['Commodity'] == 'Onion')]

# Step 2: Convert date columns to datetime format
market_data['Arrival_Date'] = pd.to_datetime(market_data['Arrival_Date'], format='%d-%m-%Y')
weather_data['Date'] = pd.to_datetime(weather_data['Date'], format='%d-%m-%Y')

# Step 3: Merge the datasets on date
merged_data = pd.merge(market_data, weather_data, left_on='Arrival_Date', right_on='Date', how='inner')
merged_data.drop('Date', axis=1, inplace=True)  # Drop redundant date column

# Step 4: Set index to date and handle missing values with interpolation
merged_data.set_index('Arrival_Date', inplace=True)
merged_data.interpolate(method='time', inplace=True)

# Step 5: Normalize numerical features and the target
feature_cols = ['Humidity', 'PRECTOTCORR', 'PS', 'WS2M', 'GWETTOP', 'ALLSKY_SFC_LW_DWN', 
                'ALLSKY_SFC_SW_DNI', 'Temperature', 'TS', 'WD10M']
feature_scaler = MinMaxScaler()
merged_data[feature_cols] = feature_scaler.fit_transform(merged_data[feature_cols])

target_scaler = MinMaxScaler()
merged_data['Modal_Price'] = target_scaler.fit_transform(merged_data[['Modal_Price']])

# Step 6: Feature engineering
# Add lagged prices and moving average
merged_data['Price_Lag1'] = merged_data['Modal_Price'].shift(1)
merged_data['Price_Lag7'] = merged_data['Modal_Price'].shift(7)
merged_data['Price_MA7'] = merged_data['Modal_Price'].rolling(window=7, min_periods=1).mean()

# Add month as a feature for seasonality
merged_data['Month'] = merged_data.index.month

# One-hot encode categorical variables (if present, e.g., Variety or Grade)
if 'Variety' in merged_data.columns:
    merged_data = pd.get_dummies(merged_data, columns=['Variety', 'Grade'], drop_first=True)

# Drop any remaining NaN values after feature engineering
merged_data.dropna(inplace=True)

# Step 7: Split data into training and validation sets
# Use data before 2020 for training, 2020 and later for validation
train_data = merged_data[merged_data.index < '2020-01-01']
val_data = merged_data[merged_data.index >= '2020-01-01']

# Step 8: Define all features for modeling
all_features = feature_cols + ['Price_Lag1', 'Price_Lag7', 'Price_MA7', 'Month'] + \
               [col for col in merged_data.columns if 'Variety_' in col or 'Grade_' in col]

# Step 9: Create sequences for LSTM
def create_sequences(data, seq_length, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][feature_cols].values)
        y.append(data.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

seq_length = 30  # Use past 30 days to predict the next day
X_train_lstm, y_train_lstm = create_sequences(train_data, seq_length, all_features, 'Modal_Price')
X_val_lstm, y_val_lstm = create_sequences(val_data, seq_length, all_features, 'Modal_Price')

# Step 10: Build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(all_features))))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_val_lstm, y_val_lstm))

# Step 11: Make predictions with LSTM and inverse transform
y_pred_lstm = lstm_model.predict(X_val_lstm)
y_pred_lstm_inv = target_scaler.inverse_transform(y_pred_lstm)
y_val_lstm_inv = target_scaler.inverse_transform(y_val_lstm.reshape(-1, 1))

# Step 12: Evaluate LSTM model
rmse_lstm = np.sqrt(mean_squared_error(y_val_lstm_inv, y_pred_lstm_inv))
mape_lstm = np.mean(np.abs((y_val_lstm_inv - y_pred_lstm_inv) / y_val_lstm_inv)) * 100
r2_lstm = r2_score(y_val_lstm_inv, y_pred_lstm_inv)
print(f"LSTM - RMSE: {rmse_lstm}, MAPE: {mape_lstm}%, R²: {r2_lstm}")

# Step 13: Prepare data for XGBoost (no sequences needed)
X_train_xgb = train_data[all_features]
y_train_xgb = train_data['Modal_Price']
X_val_xgb = val_data[all_features]
y_val_xgb = val_data['Modal_Price']

# Step 14: Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_xgb, y_train_xgb)

# Step 15: Make predictions with XGBoost and inverse transform
y_pred_xgb = xgb_model.predict(X_val_xgb)
y_pred_xgb_inv = target_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_val_xgb_inv = target_scaler.inverse_transform(y_val_xgb.values.reshape(-1, 1))

# Step 16: Evaluate XGBoost model
rmse_xgb = np.sqrt(mean_squared_error(y_val_xgb_inv, y_pred_xgb_inv))
mape_xgb = np.mean(np.abs((y_val_xgb_inv - y_pred_xgb_inv) / y_val_xgb_inv)) * 100
r2_xgb = r2_score(y_val_xgb_inv, y_pred_xgb_inv)
print(f"XGBoost - RMSE: {rmse_xgb}, MAPE: {mape_xgb}%, R²: {r2_xgb}")

# Step 17: Visualize actual vs. predicted prices for LSTM
plt.figure(figsize=(12, 6))
plt.plot(val_data.index[seq_length:], y_val_lstm_inv, label='Actual')
plt.plot(val_data.index[seq_length:], y_pred_lstm_inv, label='Predicted (LSTM)')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.legend()
plt.title('LSTM: Actual vs. Predicted Modal Prices')
plt.show()

# Step 18: Visualize actual vs. predicted prices for XGBoost
plt.figure(figsize=(12, 6))
plt.plot(val_data.index, y_val_xgb_inv, label='Actual')
plt.plot(val_data.index, y_pred_xgb_inv, label='Predicted (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.legend()
plt.title('XGBoost: Actual vs. Predicted Modal Prices')
plt.show()

# Step 19: Plot feature importances for XGBoost
importances = xgb_model.feature_importances_
feature_names = all_features
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
plt.title('XGBoost Feature Importances')
plt.show()
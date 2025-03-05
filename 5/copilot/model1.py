import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib

# Load data
data = pd.read_csv('D:/programming/Python/ML/5/cleaned_merged_market_weather1.csv', parse_dates=['Reported Date'], dayfirst=True)

# Data Preprocessing
def preprocess_data(data):
    # Extract month and create season feature
    data['Month'] = data['Reported Date'].dt.month
    data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Autumn')
    
    # One-hot encode categorical features
    categorical_features = ['State Name', 'District Name', 'Market Name', 'Variety', 'Group', 'Season']
    numerical_features = ['Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 
                          'Modal Price (Rs./Quintal)', 'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
                          'T2M', 'T2MDEW', 'PRECTOTCORR']

    # Impute missing values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    data = data.dropna(subset=['Reported Date', 'Modal Price (Rs./Quintal)'])
    X = data.drop(['Modal Price (Rs./Quintal)', 'Reported Date'], axis=1)
    y = data['Modal Price (Rs./Quintal)']

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor

X, y, preprocessor = preprocess_data(data)

# Feature Engineering
def create_lag_features(data, lag=1):
    data['Lag_Modal_Price'] = data['Modal Price (Rs./Quintal)'].shift(lag)
    return data.dropna()

data = create_lag_features(data, lag=1)

# Reshape data for LSTM
def reshape_data_for_lstm(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 3
X_lstm, y_lstm = reshape_data_for_lstm(X, y, time_steps)

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model((time_steps, X_lstm.shape[2]))
lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2)

# Save the LSTM model
lstm_model.save('lstm_model.h5')

# 2025 Prediction for LSTM
def predict_future_prices_lstm(model, preprocessor, data, start_date='2025-01-01', end_date='2025-12-31', time_steps=1):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    future_data = pd.DataFrame({'Reported Date': future_dates})
    future_data['Month'] = future_data['Reported Date'].dt.month
    future_data['Season'] = future_data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Autumn')
    
    future_data = future_data.drop(['Reported Date'], axis=1)
    future_X = preprocessor.transform(future_data)
    
    future_X_lstm = []
    for i in range(len(future_X) - time_steps):
        future_X_lstm.append(future_X[i:(i + time_steps)])
    future_X_lstm = np.array(future_X_lstm)
    
    future_prices = model.predict(future_X_lstm)
    future_data = future_data.iloc[time_steps:]
    future_data['Predicted Modal Price (Rs./Quintal)'] = future_prices
    return future_data

predicted_prices_2025_lstm = predict_future_prices_lstm(lstm_model, preprocessor, data, time_steps=time_steps)
print(predicted_prices_2025_lstm)

# Plot the predicted prices for LSTM
plt.figure(figsize=(10, 6))
plt.plot(predicted_prices_2025_lstm['Month'], predicted_prices_2025_lstm['Predicted Modal Price (Rs./Quintal)'])
plt.xlabel('Month')
plt.ylabel('Predicted Modal Price (Rs./Quintal)')
plt.title('Predicted Monthly Vegetable Prices for 2025 (LSTM)')
plt.grid(True)
plt.savefig('predicted_prices_2025_lstm.png')
plt.show()

# Save the predictions to a CSV file for LSTM
predicted_prices_2025_lstm.to_csv('predicted_prices_2025_lstm.csv', index=False)
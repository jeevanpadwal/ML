import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("D:/programming/Python/ML/Data Merging/merged_dataset.csv")  # Update with the actual dataset path

data["DATE"] = pd.to_datetime(data["DATE"])  # Ensure correct date format

data.sort_values("DATE", inplace=True)  # Ensure data is sorted by date

# Select relevant features for price prediction
features = ["T2M", "CLOUD_AMT", "WS50M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
target = "Modal_Price"

# Normalize data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
data[target] = scaler.fit_transform(data[[target]])

# Prepare time-series data for LSTM
sequence_length = 10  # Use past 10 days' data to predict the next price
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[features].iloc[i:i+sequence_length].values)
    y.append(data[target].iloc[i+sequence_length])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
epochs = 50
batch_size = 16
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# Evaluate model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict future prices
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Convert back to original scale

# Save the model
model.save("price_prediction_model.h5")
print("Model saved as price_prediction_model.h5")

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = keras.models.load_model("D:/programming/Python/ML/Dashbord/best_model_Onion.h5")

# Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature names
features = ["T2M", "CLOUD_AMT", "WS50M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
target = "Modal_Price"
sequence_length = 10

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        input_data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(input_data['weather_data'])
        
        # Ensure date column is datetime
        df["DATE"] = pd.to_datetime(df["DATE"])
        df.sort_values("DATE", inplace=True)
        
        # Normalize features
        df[features] = scaler.transform(df[features])
        
        # Prepare sequence data
        X = []
        dates = []
        
        # Create sequences from the normalized data
        for i in range(len(df) - sequence_length + 1):
            X.append(df[features].iloc[i:i+sequence_length].values)
            dates.append(df["DATE"].iloc[i+sequence_length-1])
        
        X = np.array(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Inverse transform to get actual prices
        predictions_df = pd.DataFrame(
            scaler.inverse_transform(predictions.reshape(-1, 1)),
            columns=["predicted_price"]
        )
        predictions_df["date"] = dates
        
        # Convert to list of dictionaries for JSON response
        result = predictions_df.to_dict(orient='records')
        
        return jsonify({
            "success": True,
            "predictions": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    # Get model metrics and information
    return jsonify({
        "model_name": "LSTM Price Predictor",
        "features": features,
        "sequence_length": sequence_length,
        "training_loss": 0.00028194,
        "validation_loss": 0.00035142,
        "test_loss": 0.00035142,
        "last_updated": "2025-03-01"
    })

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        # Get data from request
        input_data = request.get_json()
        days = input_data.get('days', 7)
        
        # Get latest sequence data
        latest_data = pd.DataFrame(input_data['latest_data'])
        latest_data["DATE"] = pd.to_datetime(latest_data["DATE"])
        latest_data.sort_values("DATE", inplace=True)
        
        # Ensure we have enough data for a sequence
        if len(latest_data) < sequence_length:
            return jsonify({
                "success": False,
                "error": f"Need at least {sequence_length} days of data for forecasting"
            }), 400
        
        # Normalize features
        latest_data[features] = scaler.transform(latest_data[features])
        
        # Get the most recent sequence
        latest_sequence = latest_data[features].iloc[-sequence_length:].values.reshape(1, sequence_length, len(features))
        
        # Initialize results
        forecast_results = []
        current_sequence = latest_sequence.copy()
        
        # Forecast for the requested number of days
        last_date = latest_data["DATE"].iloc[-1]
        
        for i in range(days):
            # Make prediction for next day
            next_pred = model.predict(current_sequence)
            
            # Calculate next date
            next_date = last_date + datetime.timedelta(days=i+1)
            
            # Store the prediction
            forecast_results.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_price": float(scaler.inverse_transform(next_pred.reshape(-1, 1))[0][0])
            })
            
            # Update sequence for next prediction (remove oldest, add newest)
            next_sequence = np.append(current_sequence[:, 1:, :], 
                                      np.array([[[0] * len(features)]]), 
                                      axis=1)
            
            # We only have the prediction for the target variable
            # For simplicity, we're leaving the weather features as zeros
            # In a real application, you'd use actual weather forecasts here
            
            current_sequence = next_sequence
            
        return jsonify({
            "success": True,
            "forecast": forecast_results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
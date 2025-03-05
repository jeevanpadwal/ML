import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import matplotlib.pyplot as plt

def train_prophet_model(commodity_df, weather_features=None, district=None):
    """
    Train Facebook Prophet model for a specific commodity and district
    
    Parameters:
    -----------
    commodity_df : DataFrame
        DataFrame containing data for a specific commodity
    weather_features : list
        List of weather feature columns to include as regressors
    district : str
        If provided, filter data to specific district
        
    Returns:
    --------
    model : Prophet model
        Trained Prophet model
    forecast : DataFrame
        Forecast results
    metrics : dict
        Performance metrics
    """
    # Filter by district if specified
    if district:
        df = commodity_df[commodity_df['District'] == district].copy()
    else:
        df = commodity_df.copy()
    
    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = df[['Arrival_Date', 'Modal_Price']].rename(
        columns={'Arrival_Date': 'ds', 'Modal_Price': 'y'}
    )
    
    # Initialize model with appropriate seasonality settings
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Agricultural prices often have multiplicative seasonality
        changepoint_prior_scale=0.05,       # Flexibility in trend changes
        seasonality_prior_scale=10.0        # Stronger seasonality component
    )
    
    # Add monthly seasonality (important for agricultural cycles)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add weather regressors if provided
    if weather_features and all(feat in df.columns for feat in weather_features):
        for feature in weather_features:
            prophet_df[feature] = df[feature]
            model.add_regressor(feature)
    
    # Fit the model
    model.fit(prophet_df)
    
    # Create future dataframe for forecasting (30 days into future)
    future = model.make_future_dataframe(periods=30)
    
    # Add regressor values to future dataframe
    if weather_features and all(feat in df.columns for feat in weather_features):
        # For simplicity, use last known values for future forecast
        # In production, you would use actual weather forecasts
        for feature in weather_features:
            future[feature] = future['ds'].map(
                lambda x: df[df['Arrival_Date'] <= x][feature].iloc[-1] 
                if x <= df['Arrival_Date'].max() 
                else df[feature].iloc[-1]
            )
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Evaluate with cross-validation
    cv_results = cross_validation(
        model=model,
        initial='365 days',        # Use 1 year of data as minimum
        period='30 days',          # Test on 30 days
        horizon='30 days',         # Forecast 30 days ahead
        parallel='threads'
    )
    
    # Calculate performance metrics
    metrics = performance_metrics(cv_results)
    
    # Create summary metrics
    summary_metrics = {
        'rmse': metrics['rmse'].mean(),
        'mape': metrics['mape'].mean() * 100,  # Convert to percentage
        'mae': metrics['mae'].mean()
    }
    
    return model, forecast, summary_metrics

def train_lstm_model(commodity_df, sequence_length=30):
    """
    Train LSTM model for price prediction
    
    Parameters:
    -----------
    commodity_df : DataFrame
        DataFrame containing data for a specific commodity
    sequence_length : int
        Number of time steps to use for LSTM input sequence
        
    Returns:
    --------
    model : Keras model
        Trained LSTM model
    history : History object
        Training history
    metrics : dict
        Performance metrics
    """
    # Sort by date
    df = commodity_df.sort_values('Arrival_Date')
    
    # Select relevant features
    price_cols = ['Modal_Price', 'Min_Price', 'Max_Price']
    weather_cols = [col for col in df.columns if any(w in col for w in ['Temperature', 'Rainfall', 'Humidity'])]
    lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    time_cols = ['Month', 'Quarter', 'DayOfWeek']
    
    # Combine features
    features = price_cols + weather_cols + lag_cols
    
    # Add one-hot encoded time features
    df_encoded = pd.get_dummies(df[time_cols])
    feature_cols = features + list(df_encoded.columns)
    
    # Combine all features
    X_data = pd.concat([df[features], df_encoded], axis=1)
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(df[['Modal_Price']])
    
    # Create sequences for LSTM
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        # Target is the price 30 days in the future
        target_idx = min(i + sequence_length + 30, len(y_scaled) - 1)
        y_sequences.append(y_scaled[target_idx])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split into train/test using time series split
    train_size = int(len(X_sequences) * 0.8)
    X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
    y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(sequence_length, X_sequences.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions on test set
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get actual prices
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = np.mean(np.abs(y_test_actual - y_pred))
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return model, history, metrics

def train_ensemble_model(commodity_df, forecast_horizon=30):
    """
    Train an ensemble model that combines Prophet and LSTM predictions
    """
    # Split data for training and validation (time-based)
    train_size = int(len(commodity_df) * 0.8)
    train_df = commodity_df.iloc[:train_size]
    val_df = commodity_df.iloc[train_size:]
    
    # Train Prophet model
    weather_features = ['Temperature', 'Rainfall', 'Humidity', 
                        'Temperature_seasonal_deviation', 'Rainfall_seasonal_deviation']
    prophet_model, prophet_forecast, prophet_metrics = train_prophet_model(
        train_df, weather_features=weather_features
    )
    
    # Train LSTM model
    lstm_model, lstm_history, lstm_metrics = train_lstm_model(train_df)
    
    # Generate predictions for validation set
    
    # For Prophet
    future_prophet = prophet_model.make_future_dataframe(periods=forecast_horizon)
    for feature in weather_features:
        # Use actual weather data for validation period
        for i, date in enumerate(future_prophet['ds']):
            if date in val_df['Arrival_Date'].values:
                future_prophet.loc[i, feature] = val_df[val_df['Arrival_Date'] == date][feature].values[0]
            else:
                # Use last known value
                future_prophet.loc[i, feature] = train_df[feature].iloc[-1]
    
    prophet_preds = prophet_model.predict(future_prophet)
    prophet_val_preds = prophet_preds[prophet_preds['ds'].isin(val_df['Arrival_Date'])]['yhat'].values
    
    # For LSTM, we need to prepare sequences
    # (simplified for brevity - in a full implementation, this would match the training preparation)
    lstm_val_preds = np.zeros(len(val_df))  # Placeholder
    
    # Combine predictions with a simple meta-model (XGBoost)
    from xgboost import XGBRegressor
    
    # Prepare meta-features
    meta_features = np.column_stack([
        prophet_val_preds,
        lstm_val_preds,
        val_df['Month'].values,
        val_df['Temperature'].values,
        val_df['Rainfall'].values
    ])
    
    # Train meta-model
    meta_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    meta_model.fit(meta_features, val_df['Modal_Price'].values)
    
    # Final predictions
    final_preds = meta_model.predict(meta_features)
    
    # Calculate ensemble metrics
    ensemble_rmse = np.sqrt(mean_squared_error(val_df['Modal_Price'].values, final_preds))
    ensemble_mape = mean_absolute_percentage_error(val_df['Modal_Price'].values, final_preds) * 100
    
    ensemble_metrics = {
        'rmse': ensemble_rmse,
        'mape': ensemble_mape,
        'prophet_rmse': prophet_metrics['rmse'],
        'prophet_mape': prophet_metrics['mape'],
        'lstm_rmse': lstm_metrics['rmse'],
        'lstm_mape': lstm_metrics['mape']
    }
    
    # Create ensemble model object to return
    ensemble = {
        'prophet_model': prophet_model,
        'lstm_model': lstm_model,
        'meta_model': meta_model,
        'metrics': ensemble_metrics
    }
    
    return ensemble

def evaluate_model_performance(model, test_data, commodity):
    """
    Evaluate model on test data
    """
    # Get actual prices
    actual_prices = test_data['Modal_Price'].values
    
    # Get predictions (implementation depends on model type)
    if isinstance(model, dict) and 'prophet_model' in model:  # Ensemble
        # Implementation for ensemble prediction
        prophet_model = model['prophet_model']
        lstm_model = model['lstm_model']
        meta_model = model['meta_model']
        
        # Get Prophet predictions
        future = prophet_model.make_future_dataframe(periods=0)
        prophet_forecast = prophet_model.predict(future)
        prophet_preds = prophet_forecast['yhat'].values[-len(actual_prices):]
        
        # Get LSTM predictions (simplified)
        lstm_preds = np.zeros_like(actual_prices)  # Placeholder
        
        # Combine with meta-model
        meta_features = np.column_stack([
            prophet_preds,
            lstm_preds,
            test_data['Month'].values,
            test_data['Temperature'].values,
            test_data['Rainfall'].values
        ])
        
        predicted_prices = meta_model.predict(meta_features)
    else:
        # Single model prediction (Prophet)
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        predicted_prices = forecast['yhat'].values[-len(actual_prices):]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    
    print(f"Performance metrics for {commodity}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'actual': actual_prices,
        'predicted': predicted_prices
    }

def recommend_crops(models, district, current_date, n_recommendations=3):
    """
    Recommend most profitable crops to grow based on model predictions
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models for each commodity
    district : str
        District to make recommendations for
    current_date : datetime
        Current date for prediction
    n_recommendations : int
        Number of recommendations to make
        
    Returns:
    --------
    recommendations : list
        List of recommended crops with expected prices
    """
    from datetime import timedelta
    
    # Calculate typical growing season (3 months)
    harvest_date = current_date + timedelta(days=90)
    
    # Get predictions for each crop at harvest time
    crop_predictions = {}
    
    for commodity, model in models.items():
        # Make future dataframe for Prophet
        if isinstance(model, dict) and 'prophet_model' in model:  # Ensemble
            prophet_model = model['prophet_model']
            future = prophet_model.make_future_dataframe(periods=90)  # 90 days ahead
            forecast = prophet_model.predict(future)
            
            # Get prediction for harvest date
            harvest_prediction = forecast[forecast['ds'] == harvest_date]['yhat'].values
            
            if len(harvest_prediction) > 0:
                predicted_price = harvest_prediction[0]
            else:
                # Use last prediction if harvest date not in forecast
                predicted_price = forecast['yhat'].iloc[-1]
        else:
            # Single Prophet model
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)
            
            # Get prediction for harvest date
            harvest_prediction = forecast[forecast['ds'] == harvest_date]['yhat'].values
            
            if len(harvest_prediction) > 0:
                predicted_price = harvest_prediction[0]
            else:
                predicted_price = forecast['yhat'].iloc[-1]
        
        # Store prediction
        crop_predictions[commodity] = predicted_price
    
    # Sort crops by predicted price
    sorted_crops = sorted(crop_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N crops
    recommendations = []
    for i, (crop, price) in enumerate(sorted_crops[:n_recommendations]):
        recommendations.append({
            'rank': i + 1,
            'crop': crop,
            'predicted_price': price,
            'harvest_date': harvest_date
        })
    
    return recommendations
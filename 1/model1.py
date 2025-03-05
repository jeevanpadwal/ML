import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For time series processing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# For advanced modeling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Advanced ML models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Neural Networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd

# Function to read and prepare data
import pandas as pd

# Function to read and prepare data
def load_and_prepare_data(file_path):
    """Load and prepare the agricultural dataset"""
    
    print("Loading dataset...")
    try:
        # Try normal load first
        df = pd.read_csv(file_path)
    except MemoryError:
        # If memory error, try loading in chunks
        chunk_list = []
        for chunk in pd.read_csv(file_path, chunksize=10000):
            chunk_list.append(chunk)
        df = pd.concat(chunk_list)

    # Convert date column to datetime
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True)

    # Sort by date
    df = df.sort_values('Arrival_Date')

    # Check and handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(method='ffill')  # Forward fill for time series data
    df = df.dropna()  # Drop any remaining rows with NaN
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Create additional time features
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
    df['Quarter'] = df['Arrival_Date'].dt.quarter

    # Ensure only numeric columns are considered for filtering
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

    # Drop rows where any numeric column has a value greater than 10,000
    threshold = 10000
    initial_rows = df.shape[0]
    mask = (numeric_df > threshold).any(axis=1)  # Identify rows to drop
    df = df[~mask]  # Keep rows where condition is False
    final_rows = df.shape[0]
    dropped_rows = initial_rows - final_rows

    print(f"Rows before dropping high values: {initial_rows}")
    print(f"Rows after dropping high values: {final_rows}")
    print(f"Total rows dropped: {dropped_rows}")

    return df



# Function to perform exploratory data analysis
def perform_eda(df):
    """Perform memory-efficient EDA on agricultural data"""
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    # Summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Memory-efficient approach to histograms
    plt.figure(figsize=(15, 6))
    
    # Calculate histogram data manually instead of letting seaborn do it
    # This prevents seaborn from consuming too much memory
    for i, col in enumerate(['Min_Price', 'Max_Price', 'Modal_Price']):
        plt.subplot(1, 3, i+1)
        # Drop nulls and compute histogram values manually
        values = df[col].dropna().values
        hist, bin_edges = np.histogram(values, bins=50)
        # Plot using matplotlib directly
        plt.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), alpha=0.7)
        # Add a density curve using a downsampled version
        if len(values) > 10000:
            # Randomly sample for KDE calculation
            sample_idx = np.random.choice(len(values), size=10000, replace=False)
            sample = values[sample_idx]
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(sample)
            x = np.linspace(min(bin_edges), max(bin_edges), 200)
            plt.plot(x, kde(x) * len(values) * (bin_edges[1]-bin_edges[0]), 'r-')
        plt.title(f'{col} Distribution')
    
    plt.tight_layout()
    plt.savefig('price_distributions.png')
    plt.close()
    
    # For time series, plot with downsampling
    plt.figure(figsize=(15, 6))
    # Use efficient downsampling for time series plot
    if len(df) > 10000:
        # Plot every nth point
        nth_point = len(df) // 10000 + 1
        plot_df = df.iloc[::nth_point]
    else:
        plot_df = df
    plt.plot(plot_df['Arrival_Date'], plot_df['Modal_Price'])
    plt.title('Modal Price Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Modal Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('modal_price_trend.png')
    plt.close()
    
    # Correlation analysis - this is memory efficient
    weather_cols = ['Pune_QV2M', 'Pune_PRECTOTCORR', 'Pune_PS', 'Pune_WS2M', 
                   'Pune_GWETTOP', 'Pune_ALLSKY_SFC_LW_DWN', 
                   'Pune_ALLSKY_SFC_SW_DNI', 'Pune_T2M', 'Pune_TS', 'Pune_WD10M']
    
    price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
    
    # Calculate correlation in chunks if needed
    if len(df) > 100000:
        chunk_size = 100000
        n_chunks = len(df) // chunk_size + 1
        corr_sum = np.zeros((len(weather_cols + price_cols), len(weather_cols + price_cols)))
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            chunk_corr = chunk[weather_cols + price_cols].corr().values
            corr_sum += chunk_corr
        
        corr_avg = corr_sum / n_chunks
        corr_df = pd.DataFrame(corr_avg, 
                              index=weather_cols + price_cols,
                              columns=weather_cols + price_cols)
    else:
        corr_df = df[weather_cols + price_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    # Use matplotlib's imshow for better memory efficiency
    plt.imshow(corr_df.values, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title('Correlation Matrix: Weather vs. Prices')
    
    # Add correlation values as text
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            plt.text(j, i, f'{corr_df.values[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_df.values[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("\nWeather factors most correlated with Modal Price:")
    corr_with_modal = corr_df['Modal_Price'].sort_values(ascending=False)
    print(corr_with_modal[corr_with_modal.index.isin(weather_cols)])
    
    return corr_with_modal

# Function to prepare features and target
def prepare_features_target(df, target_col='Modal_Price', lag_periods=14, forecast_period=7):
    """Memory-efficient feature preparation for large datasets"""
    print(f"\nPreparing features with lag periods: {lag_periods}, forecasting ahead: {forecast_period} days")
    
    # Extract only necessary columns to save memory
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    essential_cols = ['Arrival_Date'] + [target_col]
    
    # Add weather columns (assuming these are most important based on correlation)
    weather_cols = [col for col in numerical_cols if 'Pune_' in col]
    essential_cols.extend(weather_cols)
    
    # Add basic time features
    essential_cols.extend(['Year', 'Month', 'DayOfWeek'])
    
    # Remove duplicates while preserving order
    essential_cols = list(dict.fromkeys(essential_cols))
    
    # Create a copy with only essential columns
    df_features = df[essential_cols].copy()
    
    print(f"Using {len(essential_cols)} essential columns instead of all {len(numerical_cols)} numerical columns")
    
    # Create features in batches to manage memory
    # First create the most important lag features
    for lag in range(1, min(8, lag_periods+1)):  # First focus on shorter lags (most important)
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Create rolling statistics (very important for time series)
    df_features[f'{target_col}_rolling_mean_7'] = df_features[target_col].rolling(window=7).mean().shift(1)
    df_features[f'{target_col}_rolling_std_7'] = df_features[target_col].rolling(window=7).std().shift(1)
    
    # Add weather lags more selectively (most correlated variables)
    top_weather_cols = weather_cols[:5]  # Take top 5 weather columns
    for col in top_weather_cols:
        for lag in range(1, min(5, lag_periods+1)):
            df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
    
    # Add longer lags for target if memory permits
    if lag_periods > 7:
        for lag in range(8, lag_periods+1):
            df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Create future target (what we want to predict)
    df_features[f'{target_col}_future_{forecast_period}'] = df_features[target_col].shift(-forecast_period)
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    # Set the target column
    target = df_features[f'{target_col}_future_{forecast_period}']
    
    # Remove the target and date column from features
    features = df_features.drop([f'{target_col}_future_{forecast_period}', 'Arrival_Date'], axis=1)
    
    # Store date for reference
    dates = df_features['Arrival_Date']
    
    print(f"Final features shape: {features.shape}, Target shape: {target.shape}")
    return features, target, dates
# Function to train different machine learning models
def train_ml_models(X_train, y_train, X_test, y_test):
    """Train machine learning models with memory efficiency for large datasets"""
    print("\nTraining machine learning models...")
    
    # Initialize models dictionary to store trained models
    models = {}
    results = {}
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Random Forest with memory optimizations
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Reduced number of estimators
        max_depth=15,      # Limit tree depth
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Check if dataset is very large, if so, use sample-based training
    if X_train.shape[0] > 100000:
        # Use stratified sampling based on target values
        sample_size = 100000
        # Create quantile-based bins for stratification
        y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train_scaled, y_train, 
            train_size=sample_size,
            stratify=y_bins,
            random_state=42
        )
        print(f"Training on stratified sample of {sample_size} rows")
        rf_model.fit(X_train_sample, y_train_sample)
    else:
        rf_model.fit(X_train_scaled, y_train)
    
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Model 2: XGBoost with memory optimizations
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',  # Use histogram-based algorithm for large datasets
        random_state=42,
        n_jobs=-1
    )
    
    # Use specialized XGBoost DMatrix for efficiency
    if X_train.shape[0] > 100000:
        # Train in batches
        batch_size = 50000
        n_batches = X_train.shape[0] // batch_size + 1
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_train.shape[0])
            
            if i == 0:  # First batch
                xgb_model.fit(
                    X_train_scaled[start_idx:end_idx], 
                    y_train.iloc[start_idx:end_idx],
                    verbose=False
                )
            else:  # Subsequent batches
                xgb_model.fit(
                    X_train_scaled[start_idx:end_idx], 
                    y_train.iloc[start_idx:end_idx],
                    xgb_model=xgb_model,  # Update existing model
                    verbose=False
                )
    else:
        xgb_model.fit(X_train_scaled, y_train)
    
    xgb_pred = xgb_model.predict(X_test_scaled)
    
    # Use only two models for large datasets to save memory
    models['Random Forest'] = rf_model
    models['XGBoost'] = xgb_model
    
    # Evaluate models
    results = {
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R²': r2_score(y_test, rf_pred),
            'predictions': rf_pred
        },
        'XGBoost': {
            'MAE': mean_absolute_error(y_test, xgb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'R²': r2_score(y_test, xgb_pred),
            'predictions': xgb_pred
        }
    }
    
    # Create ensemble prediction (average of the two models)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    results['Ensemble'] = {
        'MAE': mean_absolute_error(y_test, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'R²': r2_score(y_test, ensemble_pred),
        'predictions': ensemble_pred
    }
    
    # Display model performance
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R²']:.4f}")
    
    # Find best model based on R²
    best_model = max(results.items(), key=lambda x: x[1]['R²'])
    print(f"\nBest performing model: {best_model[0]} with R² of {best_model[1]['R²']:.4f}")
    
    # Feature importance (only from Random Forest to save memory)
    if hasattr(models['Random Forest'], 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importances.head(10))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importances.head(15)
        plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return models, results, scaler

# Function to train LSTM neural network
def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train LSTM with memory efficiency for large datasets"""
    print("\nTraining LSTM Neural Network...")
    
    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape y for scaling
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)
    
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
    y_test_scaled = scaler_y.transform(y_test_reshaped)
    
    # For very large datasets, use a sample for training
    if X_train_scaled.shape[0] > 50000:
        sample_size = 50000
        print(f"Using a sample of {sample_size} observations for LSTM training")
        
        # Use stratified sampling based on target values
        y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train_scaled, y_train_scaled, 
            train_size=sample_size,
            stratify=y_bins,
            random_state=42
        )
        
        X_train_scaled = X_train_sample
        y_train_scaled = y_train_sample
    
    # Reshape input data for LSTM [samples, time steps, features]
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Use a more memory-efficient LSTM architecture
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(1, X_train.shape[1])),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Use a smaller batch size and fewer epochs for large datasets
    batch_size = 64 if X_train_reshaped.shape[0] > 10000 else 32
    
    # Train the model with smaller validation split for large datasets
    history = model.fit(
        X_train_reshaped, y_train_scaled,
        epochs=50,  # Reduced epochs
        batch_size=batch_size,
        validation_split=0.1,  # Smaller validation set
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('LSTM Model MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    plt.close()
    
    # Make predictions in batches for large test sets
    if X_test_reshaped.shape[0] > 10000:
        predictions = []
        batch_size = 5000
        for i in range(0, X_test_reshaped.shape[0], batch_size):
            batch_end = min(i + batch_size, X_test_reshaped.shape[0])
            batch_preds = model.predict(X_test_reshaped[i:batch_end])
            predictions.append(batch_preds)
        y_pred_scaled = np.vstack(predictions)
    else:
        y_pred_scaled = model.predict(X_test_reshaped)
    
    # Inverse transform to get original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nLSTM Neural Network Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, y_pred, {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Function to visualize model predictions
def visualize_predictions(y_test, ml_results, lstm_results, test_dates):
    """Visualize predictions from different models"""
    print("\nVisualizing model predictions...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot actual values
    plt.plot(test_dates, y_test, 'b-', linewidth=2, label='Actual Price')
    
    # Plot ML model predictions
    colors = ['r-', 'g-', 'c-', 'm-', 'y-']
    for (model_name, metrics), color in zip(ml_results.items(), colors):
        plt.plot(test_dates, metrics['predictions'], color, alpha=0.7, linewidth=1, label=f'{model_name} Predictions')
    
    # Plot LSTM predictions
    plt.plot(test_dates, lstm_results, 'k-', linewidth=1, label='LSTM Predictions')
    
    plt.title('Model Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()

# Function to create future predictions
def make_future_predictions(df, models, scaler, lstm_model, target_col='Modal_Price', forecast_days=30):
    """Make future predictions using the trained models"""
    print(f"\nGenerating {forecast_days} days forecast...")
    
    # Get the most recent data for forecasting
    last_date = df['Arrival_Date'].max()
    print(f"Last date in dataset: {last_date}")
    
    # Create a dataframe for future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({'Arrival_Date': future_dates})
    
    # Add time features - ONLY those that were used during training
    future_df['Year'] = future_df['Arrival_Date'].dt.year
    future_df['Month'] = future_df['Arrival_Date'].dt.month
    # Remove these if they weren't in training data
    # future_df['Day'] = future_df['Arrival_Date'].dt.day
    # future_df['Quarter'] = future_df['Arrival_Date'].dt.quarter
    future_df['DayOfWeek'] = future_df['Arrival_Date'].dt.dayofweek
    
    # Important: Create lag features that were used in training
    latest_values = df.sort_values('Arrival_Date').tail(15)[target_col].values
    
    # Initialize prediction dataframe
    forecast_results = pd.DataFrame({'Date': future_dates})
    predictions = []
    
    # Loop through each day for forecasting
    for i in range(forecast_days):
        # Create a single row dataframe for this forecast day
        current_df = pd.DataFrame({'Arrival_Date': [future_dates[i]]})
        
        # Add the same time features
        current_df['Year'] = current_df['Arrival_Date'].dt.year
        current_df['Month'] = current_df['Arrival_Date'].dt.month
        current_df['DayOfWeek'] = current_df['Arrival_Date'].dt.dayofweek
        
        # Add lag features using the latest available values
        for lag in range(1, 14):  # Adjust range based on your actual lags used in training
            lag_col = f'{target_col}_lag_{lag}'
            if i < lag:
                # Use historical data if available
                current_df[lag_col] = latest_values[-lag]
            else:
                # Use previous predictions
                current_df[lag_col] = predictions[i-lag]
        
        # Prepare features for prediction
        X_future = current_df.drop('Arrival_Date', axis=1)
        
        # Make predictions with each model
        for model_name, model in models.items():
            X_future_scaled = scaler.transform(X_future)
            prediction = model.predict(X_future_scaled)[0]
            forecast_results.loc[i, f'{model_name} Forecast'] = prediction
            
        # Store this prediction for future lag features
        predictions.append(forecast_results.loc[i, f'{list(models.keys())[0]} Forecast'])
    
    # Add ensemble average
    model_cols = [col for col in forecast_results.columns if 'Forecast' in col]
    forecast_results['Ensemble Average'] = forecast_results[model_cols].mean(axis=1)
    
    # Visualization code remains the same
    plt.figure(figsize=(15, 8))
    
    historical_data = df.sort_values('Arrival_Date').tail(90)
    plt.plot(historical_data['Arrival_Date'], historical_data[target_col], 'b-', label='Historical Prices')
    
    for model_name in models.keys():
        plt.plot(forecast_results['Date'], forecast_results[f'{model_name} Forecast'], '--', alpha=0.7, label=f'{model_name} Forecast')
    
    plt.plot(forecast_results['Date'], forecast_results['Ensemble Average'], 'r-', linewidth=2, label='Ensemble Average Forecast')
    
    plt.title(f'{forecast_days}-Day Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_forecast.png')
    plt.close()
    
    return forecast_results

# Function to get price trend and farming recommendations
def generate_recommendations(forecast_df, threshold_pct=5):
    """Generate farming recommendations based on price forecasts"""
    print("\nGenerating farming recommendations based on price forecasts...")
    
    # Calculate average predicted price
    avg_price = forecast_df['Ensemble Average'].mean()
    
    # Calculate price trend (first week vs last week)
    first_week_avg = forecast_df['Ensemble Average'].head(7).mean()
    last_week_avg = forecast_df['Ensemble Average'].tail(7).mean()
    
    price_change_pct = ((last_week_avg - first_week_avg) / first_week_avg) * 100
    
    # Determine price trend
    if price_change_pct > threshold_pct:
        trend = "INCREASING"
    elif price_change_pct < -threshold_pct:
        trend = "DECREASING"
    else:
        trend = "STABLE"
    
    # Generate recommendations based on trend
    recommendations = {
        "Price Forecast": {
            "Average Expected Price": round(avg_price, 2),
            "Price Trend": trend,
            "Price Change %": round(price_change_pct, 2)
        },
        "Farming Recommendations": []
    }
    
    if trend == "INCREASING":
        recommendations["Farming Recommendations"] = [
            "Consider holding your current harvest for better prices if storage is available.",
            "Good time to consider expanding wheat cultivation area for the next season.",
            "Invest in quality storage solutions to maximize profit from increasing prices.",
            "Consider staggered selling strategy to benefit from continued price increases."
        ]
    elif trend == "DECREASING":
        recommendations["Farming Recommendations"] = [
            "Consider selling your current harvest soon to avoid further price drops.",
            "Explore value-added products to increase profit margins.",
            "Consider diversifying crops for the next season to mitigate price risk.",
            "Look for long-term storage contracts or forward selling agreements."
        ]
    else:  # STABLE
        recommendations["Farming Recommendations"] = [
            "Prices are expected to remain stable, allowing flexible selling strategy.",
            "Good time to focus on cost optimization to improve profit margins.",
            "Consider implementing quality improvements to command premium prices.",
            "Stable prices provide a good opportunity for long-term planning."
        ]
    
    # Add additional wheat-specific recommendations
    recommendations["Wheat-Specific Recommendations"] = [
        "Monitor soil moisture levels for optimal growth - current weather data suggests adjustments may be needed.",
        "Consider nitrogen application timing based on forecast weather conditions.",
        "Focus on weed control measures to maximize yield potential.",
        "Ensure proper storage conditions to maintain quality if holding harvest."
    ]
    
    # Print recommendations
    print("\nFARMING RECOMMENDATIONS:")
    print(f"Price Forecast: {recommendations['Price Forecast']}")
    print("\nGeneral Recommendations:")
    for rec in recommendations["Farming Recommendations"]:
        print(f"- {rec}")
    
    print("\nWheat-Specific Recommendations:")
    for rec in recommendations["Wheat-Specific Recommendations"]:
        print(f"- {rec}")
    
    return recommendations

# Main function to run the entire modeling pipeline
def main(data_path, target_col='Modal_Price', forecast_period=7, future_forecast_days=30):
    """Memory-efficient main function for large datasets"""
    try:
        # Configure matplotlib to use Agg backend (more memory efficient)
        import matplotlib
        matplotlib.use('Agg')
        
        # Load and prepare data
        df = load_and_prepare_data(data_path)
        
        # Free up memory
        import gc
        gc.collect()
        
        # Perform EDA
        try:
            correlations = perform_eda(df)
        except MemoryError:
            print("Memory error during EDA visualizations. Calculating correlations without plots.")
            weather_cols = ['Pune_QV2M', 'Pune_PRECTOTCORR', 'Pune_PS', 'Pune_WS2M', 
                           'Pune_GWETTOP', 'Pune_ALLSKY_SFC_LW_DWN', 
                           'Pune_ALLSKY_SFC_SW_DNI', 'Pune_T2M', 'Pune_TS', 'Pune_WD10M']
            
            price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
            
            corr_df = df[weather_cols + price_cols].corr()
            correlations = corr_df['Modal_Price'].sort_values(ascending=False)
        
        gc.collect()
        
        # Prepare features and target
        X, y, dates = prepare_features_target(df, target_col=target_col, forecast_period=forecast_period)
        
        # Free up memory by removing the original dataframe if it's very large
        if len(df) > 100000:
            # Keep a smaller copy of df for later use
            df_small = df[['Arrival_Date', target_col, 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']].copy()
            
            # Add a few key weather columns
            important_cols = correlations[correlations.index.str.startswith('Pune_')].head(5).index.tolist()
            for col in important_cols:
                if col in df.columns:
                    df_small[col] = df[col]
            
            df = df_small
            gc.collect()
        
        # Split data for training and testing
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        test_dates = dates[train_size:]
        
        print(f"Training data size: {X_train.shape}")
        print(f"Testing data size: {X_test.shape}")
        
        # Train machine learning models
        ml_models, ml_results, scaler = train_ml_models(X_train, y_train, X_test, y_test)
        gc.collect()
        
        # Train LSTM model
        lstm_model, lstm_predictions, lstm_metrics = train_lstm_model(X_train, y_train, X_test, y_test)
        gc.collect()
        
        # Visualize predictions
        visualize_predictions(y_test, ml_results, lstm_predictions, test_dates)
        gc.collect()
        
        # Make future predictions
        forecast_results = make_future_predictions(
            df, 
            ml_models, 
            scaler, 
            lstm_model, 
            target_col=target_col, 
            forecast_days=future_forecast_days
        )
        
        # Generate farming recommendations
        recommendations = generate_recommendations(forecast_results)
        
        # Save forecast results to CSV
        forecast_results.to_csv('price_forecast_results.csv', index=False)
        
        # Print final message
        print("\nForecasting completed! Results saved to CSV and visualizations saved as PNG files.")
        print("Use the recommendations to help farmers make informed decisions.")
        
        return forecast_results, recommendations
        
    except MemoryError as e:
        print(f"Memory error occurred: {e}")
        print("Try running the script on a machine with more RAM or reduce the dataset size.")
        return None, None

# Example usage with sample data path
if __name__ == "__main__":
    # This would be your actual data path
    data_path = r"D:\programming\100x-Cohort\Flask\Model\merged_market_weather_pune.csv"
    df = pd.read_csv(data_path)

    
    # Run the forecasting pipeline
    forecast_results, recommendations = main(
        data_path=data_path,
        target_col='Modal_Price',
        forecast_period=7,  # Predict prices 7 days ahead
        future_forecast_days=30  # Generate 30-day future forecast
    )
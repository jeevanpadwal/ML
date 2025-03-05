import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Attention, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import datetime

# 1. Data Loading and Initial Exploration
def load_and_explore_data(market_data_path, weather_data_path):
    """
    Load market and weather data, perform initial exploration
    """
    # Load datasets
    market_df = pd.read_csv(market_data_path, parse_dates=['Arrival_Date'], dayfirst=True)
    weather_df = pd.read_csv(weather_data_path)
    
    # Basic exploration
    print(f"Market data shape: {market_df.shape}")
    print(f"Weather data shape: {weather_df.shape}")
    
    # Check for missing values
    print("\nMissing values in market data:")
    print(market_df.isnull().sum())
    
    print("\nMissing values in weather data:")
    print(weather_df.isnull().sum())
    
    # Display basic statistics
    print("\nMarket data statistics:")
    print(market_df.describe())
    
    # Check data types
    print("\nMarket data types:")
    print(market_df.dtypes)
    
    return market_df, weather_df

# 2. Data Preprocessing
def preprocess_data(market_df, weather_df):
    """
    Preprocess and merge market and weather data
    """
    # Ensure Arrival_Date is in correct format
    if not pd.api.types.is_datetime64_any_dtype(market_df['Arrival_Date']):
        market_df['Arrival_Date'] = pd.to_datetime(market_df['Arrival_Date'], dayfirst=True)
    
    # Create date features
    market_df['DATE'] = market_df['Arrival_Date']
    market_df['YEAR'] = market_df['Arrival_Date'].dt.year
    market_df['MONTH'] = market_df['Arrival_Date'].dt.month
    market_df['DAY'] = market_df['Arrival_Date'].dt.day
    market_df['DOY'] = market_df['Arrival_Date'].dt.dayofyear
    market_df['WEEK'] = market_df['Arrival_Date'].dt.isocalendar().week
    market_df['WEEKDAY'] = market_df['Arrival_Date'].dt.dayofweek
    
    # Handle missing values in market data
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        # Fill missing values with the median of the same commodity and market
        market_df[col] = market_df.groupby(['Commodity', 'Market'])[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Process weather data
    if 'DATE' in weather_df.columns and not pd.api.types.is_datetime64_any_dtype(weather_df['DATE']):
        weather_df['DATE'] = pd.to_datetime(weather_df['DATE'], dayfirst=True)
    
    # Normalize weather parameters
    weather_features = ['WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR']
    scaler = StandardScaler()
    weather_df[weather_features] = scaler.fit_transform(weather_df[weather_features])
    
    # Merge datasets
    # Assuming weather data can be matched with market data based on date and location
    if 'State' in weather_df.columns and 'District' in weather_df.columns:
        merged_df = pd.merge(
            market_df, 
            weather_df,
            on=['DATE', 'State', 'District'],
            how='left'
        )
    else:
        # If weather data doesn't have location information, merge only on date
        merged_df = pd.merge(
            market_df, 
            weather_df,
            on=['DATE'],
            how='left'
        )
    
    # Handle any missing values after merge
    for feature in weather_features:
        if feature in merged_df.columns:
            # Fill missing weather data with the mean for that day of year
            merged_df[feature] = merged_df.groupby(['DOY'])[feature].transform(
                lambda x: x.fillna(x.mean())
            )
    
    print(f"Shape after preprocessing: {merged_df.shape}")
    return merged_df

# 3. Feature Engineering
def engineer_features(df):
    """
    Create additional features to improve model performance
    """
    # Price ratios and differences
    if all(col in df.columns for col in ['Min_Price', 'Max_Price', 'Modal_Price']):
        df['Price_Range'] = df['Max_Price'] - df['Min_Price']
        df['Price_Range_Ratio'] = df['Price_Range'] / df['Modal_Price']
    
    # Moving averages for prices (7-day and 30-day)
    df = df.sort_values(['Commodity', 'Market', 'DATE'])
    
    # Create price lags (previous 1, 3, 7 days)
    for lag in [1, 3, 7]:
        df[f'Modal_Price_Lag_{lag}'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].shift(lag)
    
    # Create rolling statistics
    for window in [7, 14, 30]:
        # Rolling mean
        df[f'Price_Rolling_Mean_{window}d'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation
        df[f'Price_Rolling_Std_{window}d'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Seasonal features
    df['Sin_Day'] = np.sin(2 * np.pi * df['DOY'] / 365)
    df['Cos_Day'] = np.cos(2 * np.pi * df['DOY'] / 365)
    
    # Weather interaction terms
    if all(col in df.columns for col in ['T2M', 'PRECTOTCORR']):
        # Temperature and precipitation interaction
        df['Temp_Precip_Interaction'] = df['T2M'] * df['PRECTOTCORR']
    
    # Supply indicators
    if 'Arrival_Qty' in df.columns:
        # Log transform arrival quantity
        df['Log_Arrival_Qty'] = np.log1p(df['Arrival_Qty'])
        
        # Rolling mean of arrival quantity
        df['Arrival_Qty_Rolling_Mean_7d'] = df.groupby(['Commodity', 'Market'])['Arrival_Qty'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Supply volatility
        df['Supply_Volatility'] = df.groupby(['Commodity', 'Market'])['Arrival_Qty'].transform(
            lambda x: x.rolling(window=14, min_periods=1).std() / x.rolling(window=14, min_periods=1).mean()
        )
    
    # One-hot encoding for categorical variables
    categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Handle missing values in engineered features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# 4. Time Series Dataset Creation
def create_sequences(data, target_col, sequence_length=30, forecast_horizon=7):
    """
    Create sequences for time series modeling
    """
    X, y = [], []
    
    # Ensure data is sorted by date
    data = data.sort_values('DATE')
    
    # Define features and target
    feature_columns = [col for col in data.columns if col != target_col 
                      and data[col].dtype in [np.float64, np.int64]]
    
    # Create sequences
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[feature_columns].iloc[i:i+sequence_length].values)
        y.append(data[target_col].iloc[i+sequence_length:i+sequence_length+forecast_horizon].values)
    
    return np.array(X), np.array(y)

# 5. Model Building
def build_lstm_model(input_shape, output_length, use_attention=True):
    """
    Build an LSTM model with optional attention mechanism
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(64, activation='tanh', return_sequences=use_attention))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Add attention mechanism if specified
    if use_attention:
        model.add(MultiHeadAttention(num_heads=4, key_dim=16))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
    
    # Output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_length, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_gru_model(input_shape, output_length):
    """
    Build a GRU model for time series forecasting
    """
    model = Sequential()
    
    # First GRU layer
    model.add(GRU(128, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Second GRU layer
    model.add(GRU(64, activation='tanh', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_length, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_transformer_model(input_shape, output_length):
    """
    Build a Transformer model for time series forecasting
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Add positional encoding (simple approach)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=input_shape[0], 
        output_dim=input_shape[1]
    )(positions)
    position_embedding = tf.tile(tf.expand_dims(position_embedding, axis=0), 
                               [tf.shape(inputs)[0], 1, 1])
    
    x = inputs + position_embedding
    
    # Multi-head attention blocks
    for _ in range(2):
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=input_shape[1]//8
        )(x, x)
        
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(input_shape[1])
        ])
        
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_length, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# 6. Training and Evaluation
def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='lstm', 
                       use_attention=True, epochs=100, batch_size=32):
    """
    Train and evaluate the specified model
    """
    # Determine input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_length = y_train.shape[1]
    
    # Build the appropriate model
    if model_type.lower() == 'lstm':
        model = build_lstm_model(input_shape, output_length, use_attention)
    elif model_type.lower() == 'gru':
        model = build_gru_model(input_shape, output_length)
    elif model_type.lower() == 'transformer':
        model = build_transformer_model(input_shape, output_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'best_{model_type}_model.h5', monitor='val_loss', 
                        save_best_only=True, verbose=1)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each forecast horizon
    metrics = {}
    for i in range(output_length):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        metrics[f'horizon_{i+1}'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    # Average metrics across all horizons
    avg_rmse = np.mean([metrics[f'horizon_{i+1}']['rmse'] for i in range(output_length)])
    avg_mae = np.mean([metrics[f'horizon_{i+1}']['mae'] for i in range(output_length)])
    avg_r2 = np.mean([metrics[f'horizon_{i+1}']['r2'] for i in range(output_length)])
    
    metrics['average'] = {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'r2': avg_r2
    }
    
    return model, history, metrics, y_pred

# 7. Visualization Functions
def plot_training_history(history):
    """
    Plot training and validation loss
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred, n_samples=5, horizon_days=7):
    """
    Plot sample predictions against actual values
    """
    plt.figure(figsize=(15, 10))
    
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        
        days = range(1, horizon_days+1)
        plt.plot(days, y_test[i], 'b-', label='Actual')
        plt.plot(days, y_pred[i], 'r--', label='Predicted')
        
        plt.title(f'Sample {i+1}: Actual vs Predicted Prices')
        plt.xlabel('Days Ahead')
        plt.ylabel('Modal Price')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_feature_importance(model, feature_names, n_features=20):
    """
    Visualize feature importance (for models that support it)
    """
    # For now, use a simple correlation-based approach
    # A more sophisticated approach would depend on the specific model type
    
    # Create a bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_names[:n_features])), feature_names[:n_features])
    plt.yticks(range(len(feature_names[:n_features])), feature_names[:n_features])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# 8. Weather Scenario Analysis
def analyze_weather_scenarios(model, base_sequence, weather_features, 
                             scenarios, feature_indices, scaler=None):
    """
    Analyze how different weather scenarios affect predicted prices
    """
    results = {}
    base_prediction = model.predict(np.array([base_sequence]))[0]
    results['base'] = base_prediction
    
    # For each scenario, modify the weather features and predict
    for scenario_name, changes in scenarios.items():
        modified_sequence = base_sequence.copy()
        
        for feature, change in changes.items():
            if feature in weather_features:
                idx = feature_indices[feature]
                
                # Apply the change to all time steps
                if scaler is not None:
                    # If we have a scaler, we need to transform the change
                    feature_idx = weather_features.index(feature)
                    scale = scaler.scale_[feature_idx]
                    modified_sequence[:, idx] += change / scale
                else:
                    modified_sequence[:, idx] += change
        
        # Predict with modified sequence
        prediction = model.predict(np.array([modified_sequence]))[0]
        results[scenario_name] = prediction
    
    return results

def plot_scenario_results(scenario_results, horizon_days=7):
    """
    Plot the results of different weather scenarios
    """
    plt.figure(figsize=(12, 6))
    
    days = range(1, horizon_days+1)
    for scenario, prediction in scenario_results.items():
        plt.plot(days, prediction, label=scenario)
    
    plt.title('Price Predictions Under Different Weather Scenarios')
    plt.xlabel('Days Ahead')
    plt.ylabel('Predicted Modal Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 9. Main Function
def main():
    """
    Main function to run the entire pipeline
    """
    # Load and explore data
    market_df, weather_df = load_and_explore_data('D:\programming\Python\ML\Data Merging\merged_dataset.csv', 'D:\programming\Python\ML\Data Merging\merged_dataset.csv')
    
    # Preprocess data
    processed_df = preprocess_data(market_df, weather_df)
    
    # Engineer features
    featured_df = engineer_features(processed_df)
    
    # Filter for specific commodity (e.g., Onion)
    onion_df = featured_df[featured_df['Commodity'] == 'Onion']
    
    # Scale the target variable
    price_scaler = MinMaxScaler()
    onion_df['Scaled_Modal_Price'] = price_scaler.fit_transform(
        onion_df[['Modal_Price']])
    
    # Create time-based split
    split_date = onion_df['DATE'].max() - pd.Timedelta(days=60)
    train_df = onion_df[onion_df['DATE'] <= split_date]
    test_df = onion_df[onion_df['DATE'] > split_date]
    
    print(f"Training data: {train_df.shape}, Test data: {test_df.shape}")
    
    # Create sequences
    sequence_length = 30  # 30 days of history
    forecast_horizon = 7  # Predict next 7 days
    
    X_train, y_train = create_sequences(
        train_df, 'Scaled_Modal_Price', sequence_length, forecast_horizon)
    X_test, y_test = create_sequences(
        test_df, 'Scaled_Modal_Price', sequence_length, forecast_horizon)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Train models
    models_to_try = ['lstm', 'gru', 'transformer']
    best_model = None
    best_metrics = None
    best_predictions = None
    best_model_name = None
    
    for model_type in models_to_try:
        print(f"\nTraining {model_type.upper()} model...")
        model, history, metrics, predictions = train_and_evaluate(
            X_train, y_train, X_test, y_test, model_type=model_type,
            use_attention=(model_type == 'lstm'))
        
        print(f"{model_type.upper()} model metrics:")
        for horizon, horizon_metrics in metrics.items():
            print(f"  {horizon}: RMSE = {horizon_metrics['rmse']:.4f}, "
                 f"MAE = {horizon_metrics['mae']:.4f}, "
                 f"R² = {horizon_metrics['r2']:.4f}")
        
        # Keep track of the best model
        if best_metrics is None or metrics['average']['rmse'] < best_metrics['average']['rmse']:
            best_model = model
            best_metrics = metrics
            best_predictions = predictions
            best_model_name = model_type
        
        # Plot training history
        plot_training_history(history)
    
    print(f"\nBest model: {best_model_name.upper()}")
    
    # Inverse transform predictions and actual values
    y_test_original = price_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    best_predictions_original = price_scaler.inverse_transform(best_predictions.reshape(-1, 1)).reshape(best_predictions.shape)
    
    # Plot sample predictions
    plot_predictions(y_test_original, best_predictions_original)
    
    # Weather scenario analysis
    weather_features = ['WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR']
    feature_columns = [col for col in onion_df.columns if col != 'Scaled_Modal_Price' 
                      and onion_df[col].dtype in [np.float64, np.int64]]
    feature_indices = {feature: feature_columns.index(feature) for feature in weather_features 
                      if feature in feature_columns}
    
    # Define weather scenarios
    scenarios = {
        'base': {},
        'heavy_rain': {'PRECTOTCORR': 2.0},  # Increased precipitation
        'drought': {'PRECTOTCORR': -1.0, 'T2M': 2.0},  # Reduced precipitation, higher temperature
        'cold_spell': {'T2M': -3.0},  # Lower temperature
        'heat_wave': {'T2M': 3.0, 'CLOUD_AMT': -0.5}  # Higher temperature, less cloud cover
    }
    
    # Get a sample sequence from test data
    sample_sequence = X_test[0]
    
    # Analyze scenarios
    scenario_results = analyze_weather_scenarios(
        best_model, sample_sequence, weather_features, 
        scenarios, feature_indices, None)
    
    # Convert predictions back to original scale
    for scenario, prediction in scenario_results.items():
        scenario_results[scenario] = price_scaler.inverse_transform(
            prediction.reshape(-1, 1)).reshape(prediction.shape)
    
    # Plot scenario results
    plot_scenario_results(scenario_results)
    
    return best_model, best_metrics, featured_df

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
def load_data():
    # Load market price data
    market_df = pd.read_csv('D:/programming/Python/ML/3/updated_dataset.csv')
    
    # Load weather data
    weather_df = pd.read_csv('D:/programming/Python/ML/3/pune_with_date.csv')
    
    return market_df, weather_df

# 2. Preprocess Market Price Data
def preprocess_market_data(market_df):
    # Filter for Pune district
    pune_df = market_df[market_df['District'] == 'Pune'].copy()
    
    # Convert 
    pune_df['Arrival_Date'] = pd.to_datetime(pune_df['Arrival_Date'])
    
    # Extract date features
    pune_df['Year'] = pune_df['Arrival_Date'].dt.year
    pune_df['Month'] = pune_df['Arrival_Date'].dt.month
    pune_df['Day'] = pune_df['Arrival_Date'].dt.day
    pune_df['DayOfYear'] = pune_df['Arrival_Date'].dt.dayofyear
    
    # Create season feature (for India's agricultural seasons)
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Summer', 4: 'Summer', 
        5: 'Summer', 6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 
        9: 'Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon', 12: 'Winter'
    }
    pune_df['Season'] = pune_df['Month'].map(season_map)
    
    # Handle missing values
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        # If all price fields are missing, drop the row
        if pune_df[col].isna().sum() > 0:
            price_cols_null = pune_df[['Min_Price', 'Max_Price', 'Modal_Price']].isna().all(axis=1)
            pune_df = pune_df[~price_cols_null].copy()
            
            # For rows with some price info, fill missing with available data
            pune_df[col] = pune_df[col].fillna(pune_df[['Min_Price', 'Max_Price', 'Modal_Price']].mean(axis=1))
    
    # Calculate price volatility (rolling 30-day standard deviation)
    commodity_groups = pune_df.groupby(['Commodity', 'Variety'])
    volatility_dfs = []
    
    for (commodity, variety), group in commodity_groups:
        group = group.sort_values('Arrival_Date')
        group['Price_Volatility'] = group['Modal_Price'].rolling(window=30, min_periods=1).std()
        volatility_dfs.append(group)
    
    pune_df = pd.concat(volatility_dfs)
    
    return pune_df

# 3. Preprocess Weather Data
def preprocess_weather_data(weather_df):
    # Create date from YEAR and DOY
    weather_df['Date'] = pd.to_datetime(weather_df['Date'],dayfirst=True)
    
    # Handle missing values with interpolation
    weather_cols = ['Humidity', 'PRECTOTCORR', 'PS', 'WS2M', 'GWETTOP', 
                   'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DNI', 'Temperature', 'TS', 'WD10M']
    
    for col in weather_cols:
        if weather_df[col].isna().sum() > 0:
            # Use spline interpolation for smoother transitions
            weather_df[col] = weather_df[col].interpolate(method='spline', order=3)
    
    # Create derived weather features
    # Temperature variation (daily range)
    if 'Temperature_Max' in weather_df.columns and 'Temperature_Min' in weather_df.columns:
        weather_df['Temp_Range'] = weather_df['Temperature_Max'] - weather_df['Temperature_Min']
    
    # Cumulative rainfall for last 7, 15, 30 days
    weather_df['Rain_7day'] = weather_df['PRECTOTCORR'].rolling(window=7, min_periods=1).sum()
    weather_df['Rain_15day'] = weather_df['PRECTOTCORR'].rolling(window=15, min_periods=1).sum()
    weather_df['Rain_30day'] = weather_df['PRECTOTCORR'].rolling(window=30, min_periods=1).sum()
    
    # Add season information
    weather_df['Month'] = weather_df['Date'].dt.month
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Summer', 4: 'Summer', 
        5: 'Summer', 6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 
        9: 'Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon', 12: 'Winter'
    }
    weather_df['Season'] = weather_df['Month'].map(season_map)
    
    return weather_df

# 4. Merge Market and Weather Data
def merge_datasets(market_df, weather_df):
    # Ensure both dataframes have comparable date columns
    market_df['Date'] = market_df['Arrival_Date'].dt.date
    weather_df['Date'] = weather_df['Date'].dt.date
    
    # Convert to datetime for consistency
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    
    # Merge on Date
    merged_df = pd.merge(market_df, weather_df, on='Date', how='left')
    
    # For any missing weather data after merge, use forward fill
    weather_cols = ['Humidity', 'PRECTOTCORR', 'PS', 'WS2M', 'GWETTOP', 
                   'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DNI', 'Temperature', 'TS', 'WD10M']
    
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(method='ffill')
    
    return merged_df

# 5. Feature Engineering
def create_features(merged_df):
    # Create lag features for price (previous 7, 14, 30 days)
    commodity_groups = merged_df.groupby(['Commodity', 'Variety'])
    lag_dfs = []
    
    for (commodity, variety), group in commodity_groups:
        group = group.sort_values('Date')
        
        # Price lags
        for lag in [7, 14, 30]:
            group[f'Price_Lag_{lag}d'] = group['Modal_Price'].shift(lag)
        
        # Price momentum (percent change)
        group['Price_7d_Momentum'] = (group['Modal_Price'] / group['Price_Lag_7d'] - 1) * 100
        group['Price_30d_Momentum'] = (group['Modal_Price'] / group['Price_Lag_30d'] - 1) * 100
        
        # Moving averages
        group['Price_MA_15d'] = group['Modal_Price'].rolling(window=15, min_periods=1).mean()
        group['Price_MA_30d'] = group['Modal_Price'].rolling(window=30, min_periods=1).mean()
        group['Price_MA_90d'] = group['Modal_Price'].rolling(window=90, min_periods=1).mean()
        
        # Add exponential moving average
        group['Price_EMA_15d'] = group['Modal_Price'].ewm(span=15, adjust=False).mean()
        
        lag_dfs.append(group)
    
    featured_df = pd.concat(lag_dfs)
        
        # Ensure required columns exist before processing
    required_cols = {'Temperature', 'Humidity', 'GWETTOP', 'WS2M', 'Season', 'Market', 'Commodity', 'Variety', 'Grade'}
    missing_cols = required_cols - set(featured_df.columns)

    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Check data preprocessing.")
    else:
        # Create weather-related features
        featured_df['Temp_7d_Change'] = (featured_df['Temperature'] - 
                                        featured_df['Temperature'].shift(7)).fillna(0)
        
        featured_df['Humid_7d_Change'] = (featured_df['Humidity'] - 
                                        featured_df['Humidity'].shift(7)).fillna(0)

        # Prevent division by zero in Grow_Index calculation
        featured_df['WS2M'] = featured_df['WS2M'].replace(0, 1e-6)  # Replace 0s with a very small value

        # Interaction between weather and growing conditions
        featured_df['Grow_Index'] = (featured_df['Temperature'].astype(float) * 
                                    featured_df['Humidity'].astype(float) * 
                                    featured_df['GWETTOP'].astype(float) / 
                                    featured_df['WS2M'].astype(float))

        # One-hot encode categorical variables safely
        categorical_cols = ['Season', 'Market', 'Commodity', 'Variety', 'Grade']
        available_cats = [col for col in categorical_cols if col in featured_df.columns]

        if available_cats:
            featured_df = pd.get_dummies(featured_df, columns=available_cats, drop_first=True)
        else:
            print("Warning: No categorical columns found for one-hot encoding.")

        # Clean up missing values from feature creation
        featured_df = featured_df.dropna().reset_index(drop=True)

    return featured_df


# 6. Prepare Data for LSTM Model
def prepare_lstm_data(df, commodity, variety, sequence_length=30):
    """Prepare LSTM input sequences for a specific commodity and variety"""
    # Filter for the specific commodity and variety
    crop_df = df[(df['Commodity'] == commodity) & (df['Variety'] == variety)].copy()
    
    # Sort by date
    crop_df = crop_df.sort_values('Date')
    
    # Select features for LSTM
    feature_cols = [
        'Modal_Price', 'Temperature', 'Humidity', 'PRECTOTCORR', 'WS2M', 'WD10M',
        'Price_Lag_7d', 'Price_Lag_14d', 'Price_Lag_30d', 'Price_7d_Momentum',
        'Price_MA_15d', 'Price_MA_30d', 'Price_Volatility', 'Grow_Index'
    ]
    
    # Ensure all needed columns exist
    for col in feature_cols:
        if col not in crop_df.columns:
            feature_cols.remove(col)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(crop_df[feature_cols])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        # Target is the next day's modal price
        y.append(scaled_data[i+sequence_length, 0])  # 0 index corresponds to Modal_Price
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, crop_df

# 7. Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    return model

# 8. XGBoost Model for Comparison
def build_xgboost_model(X_train, y_train, X_test, y_test):
    # Reshape LSTM 3D data to 2D for XGBoost
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Define model parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Build and train model
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(X_train_2d, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_2d)
    
    return xgb_model, y_pred

# 9. Prophet Model for Seasonality Analysis
def build_prophet_model(df, commodity, variety):
    # Filter and prepare data
    crop_df = df[(df['Commodity'] == commodity) & (df['Variety'] == variety)].copy()
    crop_df = crop_df.sort_values('Date')
    
    # Prophet requires 'ds' and 'y' columns
    prophet_df = pd.DataFrame({
        'ds': crop_df['Date'],
        'y': crop_df['Modal_Price']
    })
    
    # Add weather regressors
    for col in ['Temperature', 'Humidity', 'PRECTOTCORR']:
        if col in crop_df.columns:
            prophet_df[col] = crop_df[col]
    
    # Split into train and test
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    # Build model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add regressors
    for col in ['Temperature', 'Humidity', 'PRECTOTCORR']:
        if col in prophet_df.columns:
            model.add_regressor(col)
    
    # Fit model
    model.fit(train_df)
    
    # Make predictions
    forecast = model.predict(test_df)
    
    # Calculate metrics
    mse = mean_squared_error(test_df['y'], forecast['yhat'])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test_df['y'], forecast['yhat']) * 100
    
    return model, forecast, rmse, mape

# 10. Evaluate Models and Generate Insights
def evaluate_models(lstm_model, X_test, y_test, scaler, lstm_history, xgb_pred=None, prophet_metrics=None):
    # LSTM predictions
    lstm_pred = lstm_model.predict(X_test)
    
    # If scaler was provided, inverse transform the predictions
    if scaler is not None:
        # Create a dummy array of the same shape as the data that was originally scaled
        dummy = np.zeros((len(y_test), scaler.scale_.shape[0]))
        # Put the predictions in the first column
        dummy[:, 0] = lstm_pred.flatten()
        # Inverse transform
        lstm_pred_inverted = scaler.inverse_transform(dummy)[:, 0]
        
        # Do the same for actual values
        dummy = np.zeros((len(y_test), scaler.scale_.shape[0]))
        dummy[:, 0] = y_test
        y_test_inverted = scaler.inverse_transform(dummy)[:, 0]
    else:
        lstm_pred_inverted = lstm_pred.flatten()
        y_test_inverted = y_test
    
    # Calculate LSTM metrics
    lstm_mse = mean_squared_error(y_test_inverted, lstm_pred_inverted)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mape = mean_absolute_percentage_error(y_test_inverted, lstm_pred_inverted) * 100
    
    # Create comparison table of models
    models_comparison = {
        'Model': ['LSTM'],
        'RMSE': [lstm_rmse],
        'MAPE (%)': [lstm_mape]
    }
    
    # Add XGBoost if provided
    if xgb_pred is not None:
        # Transform XGBoost predictions similarly
        dummy = np.zeros((len(y_test), scaler.scale_.shape[0]))
        dummy[:, 0] = xgb_pred
        xgb_pred_inverted = scaler.inverse_transform(dummy)[:, 0]
        
        xgb_mse = mean_squared_error(y_test_inverted, xgb_pred_inverted)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mape = mean_absolute_percentage_error(y_test_inverted, xgb_pred_inverted) * 100
        
        models_comparison['Model'].append('XGBoost')
        models_comparison['RMSE'].append(xgb_rmse)
        models_comparison['MAPE (%)'].append(xgb_mape)
    
    # Add Prophet if provided
    if prophet_metrics is not None:
        prophet_rmse, prophet_mape = prophet_metrics
        models_comparison['Model'].append('Prophet')
        models_comparison['RMSE'].append(prophet_rmse)
        models_comparison['MAPE (%)'].append(prophet_mape)
    
    comparison_df = pd.DataFrame(models_comparison)
    
    return comparison_df, lstm_pred_inverted, y_test_inverted

# 11. Generate Crop Recommendations
def generate_recommendations(df, models, weather_forecast=None):
    """
    Generate crop recommendations based on predicted prices and expected profitability
    
    Args:
        df: The processed dataframe
        models: Dictionary of trained models for each crop
        weather_forecast: Optional weather forecast data
    
    Returns:
        DataFrame with crop recommendations and expected profitability
    """
    # Get unique commodities and varieties
    commodity_varieties = df[['Commodity', 'Variety']].drop_duplicates()
    
    # Placeholder for recommendations
    recommendations = []
    
    for _, row in commodity_varieties.iterrows():
        commodity = row['Commodity']
        variety = row['Variety']
        
        # Skip if no model exists for this crop
        if (commodity, variety) not in models:
            continue
        
        model_info = models[(commodity, variety)]
        
        # Get recent price trend (last 90 days)
        recent_data = df[(df['Commodity'] == commodity) & 
                        (df['Variety'] == variety)].sort_values('Date').tail(90)
        
        # Calculate expected price using the model
        if weather_forecast is not None:
            # Use weather forecast for prediction
            expected_price = model_info['predict_with_weather'](weather_forecast)
        else:
            # Use historical pattern for prediction
            expected_price = model_info['predict_future']()
        
        # Get historical price volatility
        volatility = recent_data['Price_Volatility'].mean()
        
        # Calculate seasonal suitability based on current season
        current_month = datetime.now().month
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Summer', 4: 'Summer', 
            5: 'Summer', 6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 
            9: 'Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon', 12: 'Winter'
        }
        current_season = season_map[current_month]
        
        # Calculate seasonal performance
        seasonal_data = df[(df['Commodity'] == commodity) & 
                          (df['Variety'] == variety) & 
                          (df['Season'] == current_season)]
        
        if not seasonal_data.empty:
            seasonal_performance = seasonal_data['Modal_Price'].mean()
        else:
            seasonal_performance = recent_data['Modal_Price'].mean()
        
        # Calculate expected profitability score
        # Higher score means more profitable
        profitability_score = (
            expected_price * 0.5 +  # 50% weight on expected price
            seasonal_performance * 0.3 +  # 30% weight on seasonal performance
            (1 / (volatility + 1)) * 100 * 0.2  # 20% weight on inverse volatility
        )
        
        recommendations.append({
            'Commodity': commodity,
            'Variety': variety,
            'Expected_Price': expected_price,
            'Price_Volatility': volatility,
            'Seasonal_Performance': seasonal_performance,
            'Profitability_Score': profitability_score,
            'Season': current_season
        })
    
    # Create recommendations dataframe
    recommendations_df = pd.DataFrame(recommendations)
    
    # Sort by profitability score
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values('Profitability_Score', ascending=False)
    
    return recommendations_df

# 12. Main Pipeline
def main():
    print("Loading data...")
    market_df, weather_df = load_data()
    
    print("Preprocessing market data...")
    market_df = preprocess_market_data(market_df)
    
    print("Preprocessing weather data...")
    weather_df = preprocess_weather_data(weather_df)
    
    print("Merging datasets...")
    merged_df = merge_datasets(market_df, weather_df)
    
    print("Feature engineering...")
    featured_df = create_features(merged_df)
    
    # Select top 5 commodities by volume for modeling
    top_commodities = featured_df.groupby(['Commodity', 'Variety']).size().sort_values(ascending=False).head(5)
    
    trained_models = {}
    model_comparisons = {}
    
    for (commodity, variety) in top_commodities.index:
        print(f"\nBuilding models for {commodity} - {variety}...")
        
        # Prepare data for LSTM
        X_train, X_test, y_train, y_test, scaler, crop_df = prepare_lstm_data(
            featured_df, commodity, variety, sequence_length=30
        )
        
        # Build and train LSTM model
        lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Set up early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        lstm_history = lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Build XGBoost model for comparison
        xgb_model, xgb_pred = build_xgboost_model(X_train, y_train, X_test, y_test)
        
        # Build Prophet model for seasonality analysis
        prophet_model, prophet_forecast, prophet_rmse, prophet_mape = build_prophet_model(
            featured_df, commodity, variety
        )
        
        # Evaluate models
        comparison_df, lstm_pred, actual_values = evaluate_models(
            lstm_model, X_test, y_test, scaler, lstm_history, 
            xgb_pred=xgb_pred, prophet_metrics=(prophet_rmse, prophet_mape)
        )
        
        model_comparisons[(commodity, variety)] = comparison_df
        
        # Store trained models and metadata
        trained_models[(commodity, variety)] = {
            'lstm_model': lstm_model,
            'prophet_model': prophet_model,
            'xgb_model': xgb_model,
            'scaler': scaler,
            'metrics': comparison_df,
            'predict_future': lambda: None,  # Placeholder for prediction function
            'predict_with_weather': lambda w: None  # Placeholder for weather-based prediction
        }
        
        print(f"Results for {commodity} - {variety}:")
        print(comparison_df)
    
    # Generate recommendations
    recommendations = generate_recommendations(featured_df, trained_models)
    
    print("\nTop crop recommendations for current season:")
    print(recommendations.head())
    
    return trained_models, recommendations

if __name__ == "__main__":
    trained_models, recommendations = main()
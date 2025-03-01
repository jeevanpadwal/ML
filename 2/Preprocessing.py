import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_clean_data(filepath):
    """
    Load and perform initial cleaning of agricultural market data
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'],dayfirst=True)
    
    # Sort by date
    df = df.sort_values(['Commodity', 'State', 'District', 'Market', 'Arrival_Date'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def handle_missing_values(df):
    """
    Handle missing values using appropriate techniques for time series data
    """
    # For price columns, use KNN imputation with temporal and spatial features
    price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
    
    # Check percentage of missing values
    missing_percentages = df[price_cols].isna().mean() * 100
    print(f"Missing value percentages before imputation: {missing_percentages}")
    
    # For small amounts of missing data, use KNN imputation
    if all(missing_percentages < 30):
        # Group by commodity and market for imputation
        for commodity in df['Commodity'].unique():
            for market in df[df['Commodity'] == commodity]['Market'].unique():
                mask = (df['Commodity'] == commodity) & (df['Market'] == market)
                if df[mask].shape[0] > 10:  # Only if we have enough data points
                    df.loc[mask, price_cols] = KNNImputer(n_neighbors=5).fit_transform(df.loc[mask, price_cols])
    
    # For remaining missing values, use forward fill within groups
    df[price_cols] = df.groupby(['Commodity', 'Market'])[price_cols].transform(
        lambda x: x.fillna(method='ffill')
    )
    
    # Any remaining missing values use commodity-wise median
    for col in price_cols:
        df[col] = df.groupby('Commodity')[col].transform(lambda x: x.fillna(x.median()))
    
    # Check if any missing values remain
    missing_percentages_after = df[price_cols].isna().mean() * 100
    print(f"Missing value percentages after imputation: {missing_percentages_after}")
    
    # If still missing, use global median (shouldn't happen often)
    df[price_cols] = df[price_cols].fillna(df[price_cols].median())
    
    return df

def feature_engineering(df):
    """
    Create features relevant to agricultural price prediction
    """
    # Extract date features
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
    df['Quarter'] = df['Arrival_Date'].dt.quarter
    
    # Create lag features for time series
    price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
    
    # Group by commodity and market to create lags
    for col in price_cols:
        # 7-day lag (week ago)
        df[f'{col}_7day_lag'] = df.groupby(['Commodity', 'Market'])[col].shift(7)
        
        # 30-day lag (month ago)
        df[f'{col}_30day_lag'] = df.groupby(['Commodity', 'Market'])[col].shift(30)
        
        # 90-day lag (quarter ago)
        df[f'{col}_90day_lag'] = df.groupby(['Commodity', 'Market'])[col].shift(90)
        
        # Year ago price (365-day lag) - captures annual seasonality
        df[f'{col}_365day_lag'] = df.groupby(['Commodity', 'Market'])[col].shift(365)
        
        # Rolling means
        df[f'{col}_7day_rolling_mean'] = df.groupby(['Commodity', 'Market'])[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df[f'{col}_30day_rolling_mean'] = df.groupby(['Commodity', 'Market'])[col].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
    
    # Add price volatility features (std dev)
    for col in price_cols:
        df[f'{col}_30day_volatility'] = df.groupby(['Commodity', 'Market'])[col].transform(
            lambda x: x.rolling(window=30, min_periods=5).std()
        )
    
    # Create market supply indicator (relative to average)
    # This would be better if we had quantity data, but can use price as proxy
    df['market_supply_indicator'] = df.groupby(['Commodity', 'Market', 'Month'])['Modal_Price'].transform(
        lambda x: x / x.mean()
    )
    
    return df

def merge_weather_data(df, weather_data_path):
    """
    Merge weather data with market price data
    """
    # Load weather data 
    weather_df = pd.read_csv(weather_data_path)
    weather_df['Date'] = pd.to_datetime(weather_df['Date'],dayfirst=True)
    
    # Assuming weather data has columns: Date, State, District, Temperature, Rainfall, Humidity
    
    # Resample to daily if needed
    weather_df = weather_df.groupby(['Date']).mean().reset_index()
    
    # Merge with market data
    df = pd.merge(
        df,
        weather_df,
        left_on=['Arrival_Date', 'State', 'District'],
        right_on=['Date', 'State', 'District'],
        how='left'
    )
    
    # Fill missing weather data
    weather_cols = ['Temperature', 'Humidity']
    
    # First try temporal interpolation
    df[weather_cols] = df.groupby(['State', 'District'])[weather_cols].transform(
        lambda x: x.interpolate(method='time')
    )
    
    # For any remaining NAs, use seasonal averages
    for col in weather_cols:
        df[col] = df.groupby(['State', 'District', 'Month'])[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Add weather lag features
    for col in weather_cols:
        # Previous 7 days average
        df[f'{col}_7day_avg'] = df.groupby(['State', 'District'])[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Previous 30 days average
        df[f'{col}_30day_avg'] = df.groupby(['State', 'District'])[col].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        # Deviation from seasonal norm
        df[f'{col}_seasonal_deviation'] = df[col] - df.groupby(['State', 'District', 'Month'])[col].transform('mean')
    
    return df

def create_model_ready_datasets(df, target_commodity=None, target_district=None, forecast_horizon=30):
    """
    Prepare final datasets for model training
    """
    # Filter data if specific commodity or district is requested
    if target_commodity:
        df = df[df['Commodity'] == target_commodity]
    if target_district:
        df = df[df['District'] == target_district]
    
    # Create target variables: future prices at different horizons
    for horizon in [7, 14, 30]:
        df[f'future_{horizon}d_price'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].shift(-horizon)
    
    # Create datasets for each commodity
    commodity_datasets = {}
    for commodity in df['Commodity'].unique():
        commodity_df = df[df['Commodity'] == commodity].copy()
        
        # Remove rows with NaN target (due to shifting)
        commodity_df = commodity_df.dropna(subset=[f'future_{forecast_horizon}d_price'])
        
        # Split features and target
        X = commodity_df.drop([f'future_{h}d_price' for h in [7, 14, 30]], axis=1)
        y = commodity_df[f'future_{forecast_horizon}d_price']
        
        commodity_datasets[commodity] = (X, y)
    
    return commodity_datasets

def preprocess_pipeline():
    """
    Full preprocessing pipeline
    """
    # Example usage
    data_path = "D:/programming/Python/ML/2/Pune Market Price.csv"
    weather_data_path = "D:/programming/Python/ML/2/pune_with_date.csv"
    
    # Load and clean data
    df = load_and_clean_data(data_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Merge weather data
    df = merge_weather_data(df, weather_data_path)
    
    # Create model-ready datasets
    commodity_datasets = create_model_ready_datasets(df, forecast_horizon=30)
    
    return commodity_datasets

# This would be the entry point if run as a script
if __name__ == "__main__":
    preprocess_pipeline()
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load and preprocess weather data
def preprocess_weather_data(weather_file_path):
    # Load weather data
    weather_df = pd.read_csv(weather_file_path)
    
    # Convert 'Date' to datetime - ensure dayfirst=True as specified
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
    
    # Handle missing values in weather data
    numeric_features = ['Humidity', 'PRECTOTCORR', 'PS', 'WS2M', 'GWETTOP', 
                       'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DNI', 'Temperature', 'TS']
    
    # Create imputer for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    
    # Apply imputer to numeric columns
    weather_df[numeric_features] = numeric_imputer.fit_transform(weather_df[numeric_features])
    
    # Handle missing wind direction with most common value
    if 'WD10M' in weather_df.columns and weather_df['WD10M'].isnull().any():
        weather_df['WD10M'] = weather_df['WD10M'].fillna(weather_df['WD10M'].mode()[0])
    
    # Add derived features that might be useful for agriculture
    weather_df['Temp_Range'] = weather_df['Temperature'] - weather_df['TS']
    weather_df['Rain_Day'] = weather_df['PRECTOTCORR'].apply(lambda x: 1 if x > 0 else 0)
    
    # Add district column for merging
    weather_df['District'] = 'Pune'
    
    return weather_df

# Function to load and preprocess market price data
def preprocess_market_data(market_file_path):
    # Load market data
    market_df = pd.read_csv(market_file_path)
    
    # Filter for Pune district
    pune_market_df = market_df[market_df['District'] == 'Pune']
    
    # Convert 'Arrival_Date' to datetime - ensure dayfirst=True as specified
    pune_market_df['Arrival_Date'] = pd.to_datetime(pune_market_df['Arrival_Date'], dayfirst=True)
    pune_market_df.rename(columns={'Arrival_Date': 'Date'}, inplace=True)
    
    # Handle missing price values
    price_features = ['Min_Price', 'Max_Price', 'Modal_Price']
    
    # For prices, using mean imputation by commodity and market
    for col in price_features:
        pune_market_df[col] = pune_market_df.groupby(['Commodity', 'Market'])[col].transform(
            lambda x: x.fillna(x.mean()))
    
    # If there are still NaNs (for commodities with no price history in a market)
    # use the overall mean for that commodity
    for col in price_features:
        pune_market_df[col] = pune_market_df.groupby(['Commodity'])[col].transform(
            lambda x: x.fillna(x.mean()))
    
    # If there are still any NaNs, use the median of all prices
    for col in price_features:
        if pune_market_df[col].isnull().any():
            pune_market_df[col] = pune_market_df[col].fillna(pune_market_df[col].median())
    
    return pune_market_df

# Function to merge weather and market data
def merge_datasets(weather_df, market_df):
    # Rename Date column to ensure consistent naming
    weather_df = weather_df.rename(columns={'Date': 'Date'})
    market_df = market_df.rename(columns={'Date': 'Date'})
    
    # Merge on District and Date
    merged_df = pd.merge(market_df, weather_df, on=['District', 'Date'], how='inner')
    
    return merged_df

# Function to create time features
def add_time_features(df):
    # Extract time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Calculate moving averages for prices (7-day and 30-day)
    price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
    
    # Group by commodity and market to calculate the moving averages
    for commodity in df['Commodity'].unique():
        for market in df[df['Commodity'] == commodity]['Market'].unique():
            mask = (df['Commodity'] == commodity) & (df['Market'] == market)
            
            for col in price_cols:
                # Sort by date first
                temp_df = df[mask].sort_values('Date')
                
                # Calculate moving averages
                df.loc[mask, f'{col}_7day_avg'] = temp_df[col].rolling(window=7, min_periods=1).mean().values
                df.loc[mask, f'{col}_30day_avg'] = temp_df[col].rolling(window=30, min_periods=1).mean().values
    
    # Calculate weather moving averages
    weather_cols = ['Temperature', 'Humidity', 'PRECTOTCORR']
    
    for col in weather_cols:
        # Sort by date first for accurate moving averages
        temp_df = df.sort_values('Date')
        
        # Calculate moving averages
        df[f'{col}_7day_avg'] = temp_df.groupby('District')[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df[f'{col}_30day_avg'] = temp_df.groupby('District')[col].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean())
    
    return df

# Function to encode categorical features
def encode_categorical_features(df):
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Identify categorical columns
    categorical_cols = ['Commodity', 'Variety', 'Grade', 'Market', 'Season']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Create one-hot encoding for each categorical column separately
    # to maintain column naming structure
    for col in categorical_cols:
        # Get one-hot encoding
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
        
        # Add to dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    # Keep the original categorical columns as requested
    # (Normally we would drop them, but requirement is to keep column names unchanged)
    
    return df_encoded

# Main preprocessing function that combines all steps
def preprocess_data(weather_file_path, market_file_path):
    # Process each dataset
    weather_df = preprocess_weather_data(weather_file_path)
    market_df = preprocess_market_data(market_file_path)
    
    # Merge datasets
    merged_df = merge_datasets(weather_df, market_df)
    
    # Add time-based features
    time_df = add_time_features(merged_df)
    
    # Encode categorical features
    encoded_df = encode_categorical_features(time_df)
    
    # Scale numerical features for modeling
    scaler = StandardScaler()
    numeric_cols = ['Min_Price', 'Max_Price', 'Modal_Price', 
                   'Temperature', 'Humidity', 'PRECTOTCORR', 'PS', 
                   'WS2M', 'GWETTOP', 'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DNI', 'TS']
    
    numeric_cols = [col for col in numeric_cols if col in encoded_df.columns]
    
    # Create a copy of the scaled features
    scaled_features = pd.DataFrame(
        scaler.fit_transform(encoded_df[numeric_cols]),
        columns=[f"{col}_scaled" for col in numeric_cols],
        index=encoded_df.index
    )
    
    # Add scaled features to the dataframe
    final_df = pd.concat([encoded_df, scaled_features], axis=1)
    
    return final_df, scaler
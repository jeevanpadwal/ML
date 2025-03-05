import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# ----- STEP 1: DATA LOADING -----
def load_data(file_path):
    """Load CSV data with error handling for column names and date parsing"""
    try:
        # First read the first few rows to check column names
        df_sample = pd.read_csv(file_path, nrows=5)
        
        # Check for date columns with different possible names
        date_col = None
        possible_date_cols = ['Reported Date', 'Reported_Date', 'Date', 'DATE', 'date']
        
        for col in possible_date_cols:
            if col in df_sample.columns:
                date_col = col
                break
        
        if date_col:
            # Now read the full file with proper date parsing
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            print(f"Found date column: {date_col}")
            print(f"Sample data columns: {df.columns.tolist()}")
            return df, date_col
        else:
            print(f"Warning: No recognized date column found in {file_path}")
            print(f"Available columns: {df_sample.columns.tolist()}")
            df = pd.read_csv(file_path)
            return df, None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

# Try to load the merged data first
data_path = 'merged_market_weather.csv'
if os.path.exists(data_path):
    print(f"Loading merged data from {data_path}")
    df, date_col = load_data(data_path)
else:
    print("Merged file not found, attempting to load and merge market and weather data...")
    
    # Load market data
    market_file = 'market_data.csv'  # Update with your actual file name
    market_df, market_date_col = load_data(market_file)
    
    # Load weather data
    weather_file = 'weather_data.csv'  # Update with your actual file name
    weather_df, weather_date_col = load_data(weather_file)
    
    if market_df is None or weather_df is None:
        raise ValueError("Failed to load one or both required datasets")
    
    # Standardize date column names
    if market_date_col and market_date_col != 'Reported_Date':
        market_df.rename(columns={market_date_col: 'Reported_Date'}, inplace=True)
    
    if weather_date_col and weather_date_col != 'Date':
        weather_df.rename(columns={weather_date_col: 'Date'}, inplace=True)
    
    # Convert dates to datetime
    try:
        market_df['Reported_Date'] = pd.to_datetime(market_df['Reported_Date'], errors='coerce')
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
    except KeyError as e:
        print(f"Error: Date column not found: {str(e)}")
        print(f"Market data columns: {market_df.columns.tolist()}")
        print(f"Weather data columns: {weather_df.columns.tolist()}")
        raise
    
    # Merge data
    df = pd.merge(market_df, weather_df, left_on='Reported_Date', right_on='Date', how='inner')
    date_col = 'Reported_Date'
    
    # Save the merged data
    df.to_csv('merged_market_weather.csv', index=False)
    print("Created and saved merged dataset")

# ----- STEP 2: DATA EXPLORATION AND PREPROCESSING -----
# Display basic information
print("\nDataset Info:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Ensure we have a standardized date column
if date_col and date_col != 'Reported_Date':
    df.rename(columns={date_col: 'Reported_Date'}, inplace=True)
    date_col = 'Reported_Date'

# Convert date to datetime if not already
try:
    if date_col:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"Converted {date_col} to datetime")
    else:
        # If no date column was found, try to identify it based on values
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    test_conversion = pd.to_datetime(df[col], errors='coerce')
                    if test_conversion.notna().sum() > 0.7 * len(df):  # If >70% converted successfully
                        df[col] = test_conversion
                        date_col = col
                        df.rename(columns={col: 'Reported_Date'}, inplace=True)
                        print(f"Identified and converted date column: {col}")
                        break
                except:
                    continue
except Exception as e:
    print(f"Error converting date: {str(e)}")

# If still no date column, create one based on row index for demonstration
if not date_col or 'Reported_Date' not in df.columns:
    print("Warning: No date column found or created. Creating synthetic dates for demonstration.")
    start_date = datetime(2020, 1, 1)
    df['Reported_Date'] = [start_date + timedelta(days=i) for i in range(len(df))]

# Check for missing values
print("\nMissing values in the dataset:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Identify modal price column
modal_price_col = None
possible_price_cols = ['Modal Price', 'Modal_Price', 'Price', 'modal_price', 'MODAL_PRICE']

for col in possible_price_cols:
    if col in df.columns:
        modal_price_col = col
        break

if not modal_price_col:
    # Try to identify based on column name containing "modal" and "price"
    for col in df.columns:
        if 'modal' in col.lower() and 'price' in col.lower():
            modal_price_col = col
            break

if not modal_price_col:
    print("Warning: Modal price column not found. Using the first numeric column as target.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        modal_price_col = numeric_cols[0]
    else:
        raise ValueError("No numeric columns found for modal price prediction")

print(f"Using '{modal_price_col}' as the target variable")

# Standardize the modal price column name
df.rename(columns={modal_price_col: 'Modal_Price'}, inplace=True)
modal_price_col = 'Modal_Price'

# Fill missing values
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# For categorical columns - use mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'Reported_Date' and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Extract additional time features
df['Year'] = df['Reported_Date'].dt.year
df['Month'] = df['Reported_Date'].dt.month
df['DOY'] = df['Reported_Date'].dt.dayofyear
df['Day_of_Week'] = df['Reported_Date'].dt.dayofweek

# ----- STEP 3: FEATURE ENGINEERING -----
# Create lag features for the target variable (Modal_Price)
df = df.sort_values(by='Reported_Date')

print("\nCreating time-based features...")
# Create lag features (previous days' prices)
for lag in [1, 7, 14, 30]:
    df[f'Modal_Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

# Create rolling mean features
for window in [7, 14, 30]:
    df[f'Modal_Price_Rolling_{window}'] = df['Modal_Price'].rolling(window=window, min_periods=1).mean()

# Remove rows with NaN from lag creation
df_clean = df.dropna()
print(f"Data shape after cleaning: {df_clean.shape}")

# ----- STEP 4: FEATURE SELECTION -----
# Identify weather features
weather_cols = []
weather_keyword_patterns = ['wind', 'ws', 'cloud', 'solar', 'radiation', 'temp', 't2m', 
                          'dew', 'precip', 'rain', 'humid', 'pressure']

for col in df_clean.columns:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in weather_keyword_patterns):
        weather_cols.append(col)

print(f"\nIdentified weather features: {weather_cols}")

# Identify market features - exclude date, target, lag features and weather features
market_cols = []
for col in df_clean.columns:
    if (col not in weather_cols and 
        col != 'Reported_Date' and 
        col != 'Modal_Price' and 
        not col.startswith('Modal_Price_Lag_') and 
        not col.startswith('Modal_Price_Rolling_') and
        col not in ['Year', 'Month', 'DOY', 'Day_of_Week']):
        market_cols.append(col)

print(f"Identified market features: {market_cols}")

# Select features for the model
feature_cols = []
feature_cols.extend(['Year', 'Month', 'DOY', 'Day_of_Week'])  # Time features
feature_cols.extend(market_cols)  # Market features
feature_cols.extend(weather_cols)  # Weather features
feature_cols.extend([f'Modal_Price_Lag_{lag}' for lag in [1, 7, 14, 30]])  # Lag features
feature_cols.extend([f'Modal_Price_Rolling_{window}' for window in [7, 14, 30]])  # Rolling features

print(f"\nTotal features selected: {len(feature_cols)}")

# ----- STEP 5: DATA SPLITTING -----
# Use time-based split since this is time series data
years = df_clean['Year'].unique()
print(f"Available years in the dataset: {years}")

# Determine the test year (most recent complete year)
if len(years) >= 2:
    test_year = sorted(years)[-2]  # Second last year as test
    train_years = [year for year in years if year < test_year]
    print(f"Training on years: {train_years}")
    print(f"Testing on year: {test_year}")
    
    train_data = df_clean[df_clean['Year'].isin(train_years)]
    test_data = df_clean[df_clean['Year'] == test_year]
else:
    # If we don't have enough years, do a regular 80-20 split
    print("Not enough years for time-based split, using regular train-test split")
    train_data, test_data = train_test_split(df_clean, test_size=0.2, random_state=42)

# Define target variable
target = 'Modal_Price'

X_train = train_data[feature_cols]
y_train = train_data[target]
X_test = test_data[feature_cols]
y_test = test_data[target]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ----- STEP 6: PREPROCESSING PIPELINE -----
# Identify categorical columns within feature_cols
categorical_cols = [col for col in feature_cols if col in df_clean.select_dtypes(include=['object']).columns]
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

print(f"\nCategorical features: {categorical_cols}")
print(f"Numerical features: {len(numerical_cols)}")

# Create preprocessor
if categorical_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ])

# ----- STEP 7: MODEL TRAINING -----
print("\nTraining models...")
# Define models to try
# 1. Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 2. XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
])

# Train models
print("Training Random Forest model...")
rf_pipeline.fit(X_train, y_train)

print("Training XGBoost model...")
xgb_pipeline.fit(X_train, y_train)

# ----- STEP 8: MODEL EVALUATION -----
# Evaluate on test data
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return y_pred

# Evaluate both models
y_pred_rf = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
y_pred_xgb = evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")

# Determine best model based on RMSE
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

if rf_rmse <= xgb_rmse:
    print("\nRandom Forest performed better. Using for 2025 predictions.")
    best_model = rf_pipeline
else:
    print("\nXGBoost performed better. Using for 2025 predictions.")
    best_model = xgb_pipeline

# Plot actual vs predicted prices for test data
plt.figure(figsize=(15, 7))
plt.plot(test_data['Reported_Date'], y_test, label='Actual Prices', color='blue')
plt.plot(test_data['Reported_Date'], y_pred_xgb, label='XGBoost Predicted', color='red', linestyle='--')
plt.plot(test_data['Reported_Date'], y_pred_rf, label='Random Forest Predicted', color='green', linestyle='-.')
plt.title(f'Actual vs Predicted Onion Prices for {test_year}')
plt.xlabel('Date')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted_test.png')
plt.close()
print("Saved test period evaluation plot to 'actual_vs_predicted_test.png'")

# ----- STEP 9: FORECAST FOR 2025 -----
print("\nGenerating forecasts for 2025...")
# Create date range for 2025
start_date_2025 = datetime(2025, 1, 1)
end_date_2025 = datetime(2025, 12, 31)
dates_2025 = pd.date_range(start=start_date_2025, end=end_date_2025, freq='D')

# Create a dataframe for 2025
df_2025 = pd.DataFrame({'Reported_Date': dates_2025})
df_2025['Year'] = df_2025['Reported_Date'].dt.year
df_2025['Month'] = df_2025['Reported_Date'].dt.month
df_2025['DOY'] = df_2025['Reported_Date'].dt.dayofyear
df_2025['Day_of_Week'] = df_2025['Reported_Date'].dt.dayofweek

# For categorical features, use most common values
for col in categorical_cols:
    most_common_value = df_clean[col].mode()[0]
    df_2025[col] = most_common_value
    print(f"Set {col} to '{most_common_value}' for 2025 predictions")

# For numerical market features, use the most recent values as a baseline
last_data_points = df_clean.sort_values('Reported_Date').tail(30)
for col in market_cols:
    if col not in categorical_cols and col in last_data_points.columns:
        df_2025[col] = last_data_points[col].mean()

# For weather features, use seasonal patterns from historical data
for col in weather_cols:
    if col in df_clean.columns:
        # Calculate average values by day of year
        seasonal_pattern = df_clean.groupby('DOY')[col].mean()
        # Apply seasonal pattern to 2025
        df_2025[col] = df_2025['DOY'].map(seasonal_pattern)
        # If any missing values (e.g., leap year), fill with overall mean
        if df_2025[col].isnull().any():
            df_2025[col] = df_2025[col].fillna(df_clean[col].mean())

# For lag features, we need to make predictions iteratively
# First, initialize with the last known values from historical data
last_prices = list(df_clean.sort_values('Reported_Date').tail(30)['Modal_Price'])

# Create initial lag features
for lag in [1, 7, 14, 30]:
    df_2025[f'Modal_Price_Lag_{lag}'] = np.nan

for window in [7, 14, 30]:
    df_2025[f'Modal_Price_Rolling_{window}'] = np.nan

# Make predictions for 2025
predictions_2025 = []

for i in range(len(df_2025)):
    # Update lag features
    if i == 0:
        df_2025.loc[i, 'Modal_Price_Lag_1'] = last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Lag_7'] = last_prices[-7] if len(last_prices) >= 7 else last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Lag_14'] = last_prices[-14] if len(last_prices) >= 14 else last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Lag_30'] = last_prices[-30] if len(last_prices) >= 30 else last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Rolling_7'] = np.mean(last_prices[-7:]) if len(last_prices) >= 7 else last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Rolling_14'] = np.mean(last_prices[-14:]) if len(last_prices) >= 14 else last_prices[-1]
        df_2025.loc[i, 'Modal_Price_Rolling_30'] = np.mean(last_prices[-30:]) if len(last_prices) >= 30 else last_prices[-1]
    else:
        # Use previous predictions for lag features
        if i >= 1:
            df_2025.loc[i, 'Modal_Price_Lag_1'] = predictions_2025[i-1]
        if i >= 7:
            df_2025.loc[i, 'Modal_Price_Lag_7'] = predictions_2025[i-7]
            df_2025.loc[i, 'Modal_Price_Rolling_7'] = np.mean(predictions_2025[i-7:i])
        else:
            remaining = 7 - i
            df_2025.loc[i, 'Modal_Price_Lag_7'] = last_prices[-remaining] if remaining <= len(last_prices) else last_prices[-1]
            df_2025.loc[i, 'Modal_Price_Rolling_7'] = np.mean(last_prices[-remaining:] + predictions_2025[:i]) if remaining <= len(last_prices) else np.mean(last_prices[-1:] + predictions_2025[:i])
            
        if i >= 14:
            df_2025.loc[i, 'Modal_Price_Lag_14'] = predictions_2025[i-14]
            df_2025.loc[i, 'Modal_Price_Rolling_14'] = np.mean(predictions_2025[i-14:i])
        else:
            remaining = 14 - i
            df_2025.loc[i, 'Modal_Price_Lag_14'] = last_prices[-remaining] if remaining <= len(last_prices) else last_prices[-1]
            df_2025.loc[i, 'Modal_Price_Rolling_14'] = np.mean(last_prices[-remaining:] + predictions_2025[:i]) if remaining <= len(last_prices) else np.mean(last_prices[-1:] + predictions_2025[:i])
            
        if i >= 30:
            df_2025.loc[i, 'Modal_Price_Lag_30'] = predictions_2025[i-30]
            df_2025.loc[i, 'Modal_Price_Rolling_30'] = np.mean(predictions_2025[i-30:i])
        else:
            remaining = 30 - i
            df_2025.loc[i, 'Modal_Price_Lag_30'] = last_prices[-remaining] if remaining <= len(last_prices) else last_prices[-1]
            df_2025.loc[i, 'Modal_Price_Rolling_30'] = np.mean(last_prices[-remaining:] + predictions_2025[:i]) if remaining <= len(last_prices) else np.mean(last_prices[-1:] + predictions_2025[:i])
    
    # Make prediction for current day
    try:
        X_pred = df_2025.iloc[i:i+1][feature_cols]
        pred = best_model.predict(X_pred)[0]
        predictions_2025.append(pred)
        
        # Print progress
        if i % 30 == 0:
            print(f"Generated predictions for {i+1}/{len(df_2025)} days")
    except Exception as e:
        print(f"Error predicting day {i+1}: {str(e)}")
        # Use last prediction or mean as fallback
        if predictions_2025:
            predictions_2025.append(predictions_2025[-1])
        else:
            predictions_2025.append(df_clean['Modal_Price'].mean())

# Add predictions to dataframe
df_2025['Predicted_Price'] = predictions_2025

# ----- STEP 10: VISUALIZATION OF 2025 PREDICTIONS -----
print("\nCreating visualizations...")
plt.figure(figsize=(15, 7))
plt.plot(df_2025['Reported_Date'], df_2025['Predicted_Price'], color='blue', linewidth=2)
plt.title('Predicted Onion Prices for 2025')
plt.xlabel('Date')
plt.ylabel('Modal Price (Rs./Quintal)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_prices_2025.png')
plt.close()
print("Saved 2025 daily price prediction plot to 'predicted_prices_2025.png'")

# Plot with monthly aggregation for better interpretability
monthly_prices_2025 = df_2025.groupby(df_2025['Reported_Date'].dt.month)['Predicted_Price'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 6))
bars = plt.bar(months, monthly_prices_2025, color='skyblue')
plt.plot(months, monthly_prices_2025, 'ro-', linewidth=2)

# Add value labels on top of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f'{monthly_prices_2025.iloc[i]:.0f}', 
             ha='center', va='bottom', fontweight='bold')

plt.title('Monthly Average Predicted Onion Prices for 2025')
plt.xlabel('Month')
plt.ylabel('Average Modal Price (Rs./Quintal)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_predicted_prices_2025.png')
plt.close()
print("Saved 2025 monthly average price prediction plot to 'monthly_predicted_prices_2025.png'")

# Create a heatmap to show price trends by day and month
daily_data = df_2025.copy()
daily_data['Day'] = daily_data['Reported_Date'].dt.day
daily_data['Month'] = daily_data['Reported_Date'].dt.month

# Reshape to create a month x day matrix
price_matrix = daily_data.pivot_table(index='Month', columns='Day', values='Predicted_Price', aggfunc='mean')

plt.figure(figsize=(16, 8))
sns.heatmap(price_matrix, cmap='coolwarm', annot=False, fmt=".0f")
plt.title('Onion Price Heatmap for 2025 (by Day and Month)')
plt.xlabel('Day of Month')
plt.ylabel('Month')
plt.tight_layout()
plt.savefig('onion_price_heatmap_2025.png')
plt.close()
print("Saved 2025 price heatmap to 'onion_price_heatmap_2025.png'")

# ----- STEP 11: EXPORT PREDICTIONS -----
# Save the predictions to a CSV file
df_2025[['Reported_Date', 'Predicted_Price']].to_csv('onion_price_predictions_2025.csv', index=False)
print("\nSaved predictions to 'onion_price_predictions_2025.csv'")

# Save monthly predictions for easy reference
monthly_df = pd.DataFrame({
    'Month': months,
    'Average_Price': monthly_prices_2025.values
})
monthly_df.to_csv('monthly_onion_price_predictions_2025.csv', index=False)
print("Saved monthly predictions to 'monthly_onion_price_predictions_2025.csv'")

# Calculate seasonal statistics
seasonal_stats = {
    'Winter (Dec-Feb)': df_2025[df_2025['Month'].isin([12, 1, 2])]['Predicted_Price'].mean(),
    'Spring (Mar-May)': df_2025[df_2025['Month'].isin([3, 4, 5])]['Predicted_Price'].mean(),
    'Summer (Jun-Aug)': df_2025[df_2025['Month'].isin([6, 7, 8])]['Predicted_Price'].mean(),
    'Fall (Sep-Nov)': df_2025[df_2025['Month'].isin([9, 10, 11])]['Predicted_Price'].mean()
}

print("\nSeasonal Price Predictions for 2025:")
for season, price in seasonal_stats.items():
    print(f"{season}: {price:.2f} Rs./Quintal")

# Identify price peaks and troughs
peak_month = months[monthly_prices_2025.idxmax()]
trough_month = months[monthly_prices_2025.idxmin()]
price_range = monthly_prices_2025.max() - monthly_prices_2025.min()
price_volatility = monthly_prices_2025.std()

print(f"\nPrice Volatility Analysis:")
print(f"- Highest prices expected in: {peak_month}")
print(f"- Lowest prices expected in: {trough_month}")
print(f"- Price range: {price_range:.2f} Rs./Quintal")
print(f"- Price standard deviation: {price_volatility:.2f}")

print("\nOnion price prediction for 2025 completed successfully!")
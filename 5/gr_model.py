import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Load data from CSV file
# Replace 'vegetable_market_data.csv' with the actual path to your CSV file
data = pd.read_csv('D:/programming/Python/ML/5/merged_market_weather.csv')

# Step 1: Data Preprocessing
def preprocess_data(df):
    # Convert 'Reported Date' to datetime
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')
    
    # Extract additional features from 'Reported Date'
    df['Month'] = df['Reported Date'].dt.month
    df['DayOfWeek'] = df['Reported Date'].dt.dayofweek
    
    # Fill missing YEAR and DOY if possible from Reported Date
    df['YEAR'] = df['YEAR'].fillna(df['Reported Date'].dt.year)
    df['DOY'] = df['DOY'].fillna(df['Reported Date'].dt.dayofyear)
    
    # Clean numerical columns with commas and convert to float
    numerical_cols = ['Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 
                      'Modal Price (Rs./Quintal)', 'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
                      'T2M', 'T2MDEW', 'PRECTOTCORR']
    for col in numerical_cols:
        if df[col].dtype == 'object':  # If column is string type
            df[col] = df[col].str.replace(',', '').astype(float)  # Remove commas and convert to float
        else:
            df[col] = df[col].astype(float)  # Ensure it's float if already numeric
    
    # Handle missing values in weather-related features (mean imputation)
    weather_cols = ['WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR']
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].mean())
    
    # Define features and target
    categorical_features = ['State Name', 'District Name', 'Market Name', 'Variety', 'Group']
    numerical_features = ['Arrivals (Tonnes)', 'YEAR', 'DOY', 'WS50M', 'CLOUD_AMT', 
                          'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR', 'Month', 'DayOfWeek']
    target = 'Modal Price (Rs./Quintal)'
    
    return df, categorical_features, numerical_features, target

# Preprocess the data
df_processed, categorical_features, numerical_features, target = preprocess_data(data)

# Step 2: Model Development
# Define preprocessor for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Split features and target
X = df_processed[categorical_features + numerical_features]
y = df_processed[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae:.2f} Rs./Quintal")
print(f"Root Mean Squared Error: {rmse:.2f} Rs./Quintal")
print(f"R-squared: {r2:.2f}")

# Step 3: Prediction Function
def predict_price(input_data):
    """
    Predicts Modal Price based on farmer input.
    Input: Dictionary with required features.
    Output: Predicted price and confidence interval.
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_df['Reported Date'] = pd.to_datetime(input_df['Reported Date'])
    input_df['Month'] = input_df['Reported Date'].dt.month
    input_df['DayOfWeek'] = input_df['Reported Date'].dt.dayofweek
    input_df['YEAR'] = input_df['YEAR'].fillna(input_df['Reported Date'].dt.year)
    input_df['DOY'] = input_df['DOY'].fillna(input_df['Reported Date'].dt.dayofyear)
    
    # Clean numerical columns in input
    numerical_cols = ['Arrivals (Tonnes)', 'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
                      'T2M', 'T2MDEW', 'PRECTOTCORR']
    for col in numerical_cols:
        if col in input_df.columns and isinstance(input_df[col].iloc[0], str):
            input_df[col] = input_df[col].str.replace(',', '').astype(float)
    
    # Ensure all required columns are present
    required_cols = categorical_features + numerical_features
    input_df = input_df[required_cols]
    
    # Predict price
    predicted_price = model.predict(input_df)[0]
    
    # Estimate uncertainty (using standard deviation of predictions from training)
    train_preds = model.predict(X_train)
    uncertainty = np.std(y_train - train_preds)  # Rough estimate of uncertainty
    
    return predicted_price, uncertainty

# Step 4: Farmer-Friendly Output and Yield Planning Advice
def farmer_output(input_data):
    predicted_price, uncertainty = predict_price(input_data)
    
    output = f"""
    Predicted Modal Price: {predicted_price:.2f} Rs./Quintal
    Confidence Range: {predicted_price - uncertainty:.2f} to {predicted_price + uncertainty:.2f} Rs./Quintal
    """
    
    # Yield planning advice
    avg_price = data['Modal Price (Rs./Quintal)'].mean()
    if predicted_price > avg_price * 1.2:  # High price
        advice = "The predicted price is high! Consider increasing your yield or planting more of this vegetable."
    elif predicted_price < avg_price * 0.8:  # Low price
        advice = "The predicted price is low. Consider adjusting planting schedules or exploring alternative crops."
    else:
        advice = "The predicted price is average. Maintain your current yield plan."
    
    return output + "\nAdvice: " + advice

# Example usage
sample_input =pd.read_csv("D:/programming/Python/ML/5/merged_market_weather.csv")

print(farmer_output(sample_input))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# 1. Data Preprocessing
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the vegetable price dataset
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
    
    # Extract temporal features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Create season feature
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 
        4: 'Spring', 5: 'Summer', 6: 'Summer',
        7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon', 
        10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    }
    df['Season'] = df['Month'].map(season_mapping)
    
    # Handle invalid entries
    for col in ['Modal Price', 'Temperature', 'Rainfall', 'Humidity']:
        if col in df.columns:
            # Replace invalid entries (like -999) with NaN
            df[col] = df[col].replace(-999, np.nan)
    
    print("Data preprocessing completed!")
    return df

# 2. Feature Engineering
def engineer_features(df):
    """
    Create new features and prepare data for modeling
    """
    print("Engineering features...")
    
    # Create lag features for price (previous month's price)
    df = df.sort_values(['State', 'Variety', 'Date'])
    df['Previous_Month_Price'] = df.groupby(['State', 'Variety'])['Modal Price'].shift(1)
    
    # Calculate rolling average prices (3-month window)
    df['Rolling_Avg_Price'] = df.groupby(['State', 'Market', 'Variety'])['Modal Price'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Create month-based cyclical features to capture seasonality
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    # Drop rows with NaN values after feature engineering
    df = df.dropna()
    
    print("Feature engineering completed!")
    return df

# 3. Prepare data for modeling
def prepare_for_modeling(df):
    """
    Split features and target, and prepare categorical and numerical features
    """
    print("Preparing data for modeling...")
    
    # Define feature sets
    categorical_features = ['State', 'Market', 'Variety', 'Category', 'Season']
    numerical_features = ['Temperature', 'Rainfall', 'Humidity', 'Previous_Month_Price', 
                         'Rolling_Avg_Price', 'Month_Sin', 'Month_Cos', 'Month', 'Year']
    
    # Filter columns that actually exist in the dataframe
    categorical_features = [col for col in categorical_features if col in df.columns]
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    # Split data into features and target
    X = df[categorical_features + numerical_features]
    y = df['Modal Price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data preparation completed!")
    return X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

# 4. Train and evaluate models
def train_and_evaluate_models(X_train, y_train, preprocessor):
    """
    Train different models and evaluate their performance
    """
    print("Training and evaluating models...")
    
    # Define models to evaluate
    models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Create pipelines for each model
    pipelines = {name: Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ]) for name, model in models.items()}
    
    # Evaluate models using cross-validation
    results = {}
    for name, pipeline in pipelines.items():
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        
        results[name] = {
            'MAE': -mae_scores.mean(),
            'MSE': -mse_scores.mean(),
            'RMSE': np.sqrt(-mse_scores.mean()),
            'R2': r2_scores.mean()
        }
    
    # Find the best model
    best_model_name = min(results, key=lambda x: results[x]['MAE'])
    print(f"Best model: {best_model_name}")
    print(f"Performance: {results[best_model_name]}")
    
    # Train the best model on the full training set
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, results

# 5. Create and train LSTM model
def train_lstm_model(X_train, y_train, preprocessor, numerical_features, categorical_features):
    """
    Create and train an LSTM model for time series prediction
    """
    print("Training LSTM model...")
    
    # Preprocess the training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Reshape data for LSTM (samples, time steps, features)
    # For simplicity, we'll use a time step of 1
    X_train_reshaped = X_train_transformed.reshape(X_train_transformed.shape[0], 1, X_train_transformed.shape[1])
    
    # Create LSTM model
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(1, X_train_transformed.shape[1])),
        Dropout(0.2),
        LSTM(units=32),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Create a pipeline-like object for the LSTM model
    class LSTMPipeline:
        def __init__(self, preprocessor, model):
            self.preprocessor = preprocessor
            self.model = model
            
        def predict(self, X):
            X_transformed = self.preprocessor.transform(X)
            X_reshaped = X_transformed.reshape(X_transformed.shape[0], 1, X_transformed.shape[1])
            return self.model.predict(X_reshaped).flatten()
    
    lstm_pipeline = LSTMPipeline(preprocessor, model)
    
    print("LSTM model training completed!")
    return lstm_pipeline, history

# 6. Generate 2025 predictions
def predict_2025_prices(best_model, df, preprocessor, categorical_features, numerical_features):
    """
    Generate price predictions for each month of 2025
    """
    print("Generating 2025 predictions...")
    
    # Get the last data point for each vegetable type, state, and city combination
    latest_data = df.sort_values('Date').groupby(['Variety', 'State', 'Market']).last().reset_index()
    
    # Create a template for 2025 predictions
    months = range(1, 13)
    predictions_list = []
    
    for _, row in latest_data.iterrows():
        veg_type = row['Variety']
        state = row['State']
       
        
        # Use the last known values for weather features or historical averages
        temp = row.get('Temperature', 0)
        rainfall = row.get('Rainfall', 0)
        humidity = row.get('Humidity', 0)
        
        for month in months:
            # Create a new row for each month in 2025
            new_row = row.copy()
            new_row['Month'] = month
            new_row['Year'] = 2025
            new_row['Date'] = pd.to_datetime(f'2025-{month:02d}-01')
            
            # Update season
            season_mapping = {
                1: 'Winter', 2: 'Winter', 3: 'Spring', 
                4: 'Spring', 5: 'Summer', 6: 'Summer',
                7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon', 
                10: 'Autumn', 11: 'Autumn', 12: 'Winter'
            }
            new_row['Season'] = season_mapping[month]
            
            # Update cyclical features
            new_row['Month_Sin'] = np.sin(2 * np.pi * month/12)
            new_row['Month_Cos'] = np.cos(2 * np.pi * month/12)
            
            # Add to predictions list
            predictions_list.append(new_row)
    
    # Create a DataFrame for 2025 data
    pred_df = pd.DataFrame(predictions_list)
    
    # Prepare features for prediction
    X_pred = pred_df[categorical_features + numerical_features]
    
    # Make predictions
    pred_df['Predicted_Price'] = best_model.predict(X_pred)
    
    print("2025 predictions generated!")
    return pred_df

# 7. Visualize predictions
def visualize_predictions(pred_df):
    """
    Create visualizations of the predicted prices
    """
    print("Creating visualizations...")
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Aggregate predictions by month (average across all vegetables and locations)
    monthly_avg = pred_df.groupby('Month')['Predicted_Price'].mean().reset_index()
    
    # Plot the average predicted prices for all vegetables
    plt.plot(monthly_avg['Month'], monthly_avg['Predicted_Price'], 'o-', linewidth=2, markersize=8)
    plt.title('Average Predicted Vegetable Prices for 2025', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Price per Unit', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Create a more detailed visualization by vegetable type
    plt.figure(figsize=(16, 10))
    
    # Get the top 5 vegetable types by average price
    top_veggies = pred_df.groupby('Variety')['Predicted_Price'].mean().nlargest(5).index.tolist()
    
    # Plot predictions for top vegetables
    for veg in top_veggies:
        veg_data = pred_df[pred_df['Variety'] == veg]
        monthly_veg_avg = veg_data.groupby('Month')['Predicted_Price'].mean().reset_index()
        plt.plot(monthly_veg_avg['Month'], monthly_veg_avg['Predicted_Price'], 'o-', linewidth=2, label=veg)
    
    plt.title('Predicted 2025 Prices for Top 5 Vegetables', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Price per Unit', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(fontsize=12)
    
    print("Visualizations created!")
    return monthly_avg

# 8. Main function to run the entire pipeline
def main(file_path):
    """
    Run the complete vegetable price prediction pipeline
    """
    # 1. Load and preprocess data
    print("call load and preprocess")
    df = load_and_preprocess_data(file_path)
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Prepare data for modeling
    X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features = prepare_for_modeling(df)
    
    # 4. Train and evaluate traditional models
    best_traditional_model, results = train_and_evaluate_models(X_train, y_train, preprocessor)
    
    # 5. Train LSTM model
    lstm_model, history = train_lstm_model(X_train, y_train, preprocessor, numerical_features, categorical_features)
    
    # 6. Evaluate models on test set
    traditional_pred = best_traditional_model.predict(X_test)
    traditional_mae = mean_absolute_error(y_test, traditional_pred)
    traditional_r2 = r2_score(y_test, traditional_pred)
    
    # Transform test data for LSTM
    X_test_transformed = preprocessor.transform(X_test)
    X_test_reshaped = X_test_transformed.reshape(X_test_transformed.shape[0], 1, X_test_transformed.shape[1])
    lstm_pred = lstm_model.model.predict(X_test_reshaped).flatten()
    lstm_mae = mean_absolute_error(y_test, lstm_pred)
    lstm_r2 = r2_score(y_test, lstm_pred)
    
    print("\nModel Evaluation Results:")
    print(f"Traditional Model - MAE: {traditional_mae:.2f}, R²: {traditional_r2:.2f}")
    print(f"LSTM Model - MAE: {lstm_mae:.2f}, R²: {lstm_r2:.2f}")
    
    # Select the best model based on test MAE
    if lstm_mae < traditional_mae:
        best_model = lstm_model
        print("LSTM model selected as the best model.")
    else:
        best_model = best_traditional_model
        print("Traditional model selected as the best model.")
    
    # 7. Generate predictions for 2025
    predictions_2025 = predict_2025_prices(best_model, df, preprocessor, categorical_features, numerical_features)
    
    # 8. Visualize predictions
    monthly_avg = visualize_predictions(predictions_2025)
    
    print("\nMonthly Average Predicted Prices for 2025:")
    for _, row in monthly_avg.iterrows():
        month_name = datetime(2025, int(row['Month']), 1).strftime('%B')
        price = row['Predicted_Price']
        print(f"{month_name}: ₹{price:.2f}")
    
    return predictions_2025, monthly_avg, best_model

#Example usage (would need actual file path to run)
predictions, monthly_averages, model = main("D:/programming/Python/ML/7/cleaned_merged_market_weather1.csv")
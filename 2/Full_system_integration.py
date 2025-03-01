import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import matplotlib.pyplot as plt
import json

# Import our custom modules
# In production, these would be properly packaged
from preprocess import load_and_clean_data, handle_missing_values, feature_engineering, merge_weather_data, create_model_ready_datasets
from models import train_prophet_model, train_lstm_model, train_ensemble_model, evaluate_model_performance, recommend_crops

def main():
    """
    Main function to run the entire pipeline
    """
    print("Starting agricultural price prediction system...")
    
    # Step 1: Load and preprocess data
    data_path = "agricultural_market_data.csv" 
    weather_data_path = "weather_data.csv"
    
    # Load data
    print("Loading and cleaning data...")
    df = load_and_clean_data(data_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    print("Performing feature engineering...")
    df = feature_engineering(df)
    
    # Merge weather data
    print("Merging weather data...")
    df = merge_weather_data(df, weather_data_path)
    
    # Step 2: Train models for each commodity
    print("Creating model datasets...")
    commodity_datasets = create_model_ready_datasets(df, forecast_horizon=30)
    
    # Define top commodities to model
    # In a full implementation, we'd model all commodities
    top_commodities = [
        'Rice', 'Wheat', 'Maize', 'Potato', 'Onion', 
        'Tomato', 'Cotton', 'Sugarcane', 'Soybean'
    ]
    
    # Filter to only include commodities we actually have in our data
    available_commodities = [c for c in top_commodities if c in commodity_datasets.keys()]
    
    # Create output directory for models
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Train models for each commodity
    model_performances = {}
    trained_models = {}
    
    for commodity in available_commodities:
        print(f"Training models for {commodity}...")
        X, y = commodity_datasets[commodity]
        
        # Split data into train and test (keeping chronological order)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        train_df = X_train.copy()
        train_df['Modal_Price'] = y_train
        
        test_df = X_test.copy()
        test_df['Modal_Price'] = y_test
        
        # Train ensemble model
        model = train_ensemble_model(train_df)
        
        # Evaluate model
        performance = evaluate_model_performance(model, test_df, commodity)
        
        # Save metrics
        model_performances[commodity] = {
            'rmse': float(performance['rmse']),
            'mape': float(performance['mape']),
            'mae': float(performance['mae'])
        }
        
        # Save model
        trained_models[commodity] = model
        joblib.dump(model, f'models/{commodity}_model.pkl')
        
        # Save model performance
        with open(f'results/{commodity}_performance.json', 'w') as f:
            json.dump(model_performances[commodity], f)
        
        # Plot predictions vs actual for visualization
        plt.figure(figsize=(12, 6))
        plt.plot(performance['actual'], label='Actual')
        plt.plot(performance['predicted'], label='Predicted')
        plt.title(f'{commodity} Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.savefig(f'results/{commodity}_prediction.png')
        plt.close()
    
    # Step 3: Calculate overall performance
    overall_rmse = np.mean([perf['rmse'] for perf in model_performances.values()])
    overall_mape = np.mean([perf['mape'] for perf in model_performances.values()])
    
    print("\nOverall Model Performance:")
    print(f"Average RMSE: {overall_rmse:.2f}")
    print(f"Average MAPE: {overall_mape:.2f}%")
    
    # Step 4: Generate crop recommendations for each district
    print("\nGenerating crop recommendations...")
    
    # Get list of districts
    districts = df['District'].unique()
    
    recommendations_by_district = {}
    
    # Current date for recommendation
    current_date = datetime.now()
    
    # For each district, recommend crops
    for district in districts:
        recommendations = recommend_crops(
            trained_models, 
            district, 
            current_date,
            n_recommendations=3
        )
        
        recommendations_by_district[district] = recommendations
    
    # Save recommendations
    with open('results/crop_recommendations.json', 'w') as f:
        json.dump(recommendations_by_district, f)
    
    print("Crop recommendations generated and saved.")
    
    # Step 5: Create API endpoints for the frontend
    # In a production system, we would implement Flask or FastAPI endpoints here
    # For simplicity, we'll just outline the structure
    
    print("""
    System successfully trained and evaluated!
    
    To implement the full system:
    1. Set up API endpoints for the frontend
    2. Connect the React dashboard to these endpoints
    3. Deploy the model with scheduled retraining
    4. Implement user authentication for farmers
    
    The system is ready for integration with the frontend dashboard.
    """)

if __name__ == "__main__":
    main()
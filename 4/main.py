import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Import our custom modules
from preprocessing import preprocess_data
from time_series_model import CropPricePredictor
from seasonality_analysis import CropAnalyzer

# For simplicity, we're assuming these modules are in the same directory
# In a real implementation, you would import them as shown above

def main():
    """
    Main function to orchestrate the crop recommendation system
    """
    print("=" * 80)
    print("Crop Recommendation System for Pune Farmers".center(80))
    print("=" * 80)
    
    # Define paths to data files
    # In a real implementation, these would be command-line arguments or config settings
    weather_file = "D:/programming/Python/ML/4/pune_with_date.csv"
    market_file = "D:/programming/Python/ML/4/updated_dataset.csv"
    
    # Check if files exist
    if not os.path.exists(weather_file) or not os.path.exists(market_file):
        print(f"Error: Data files not found. Please ensure {weather_file} and {market_file} exist.")
        return
    
    print("\nStep 1: Data Preprocessing")
    print("-" * 80)
    
    # Preprocess data
    try:
        print("Loading and preprocessing data...")
        processed_df, scaler = preprocess_data(weather_file, market_file)
        print(f"Successfully processed data. Shape: {processed_df.shape}")
        
        # Sample of preprocessed data
        print("\nSample of preprocessed data:")
        print(processed_df[['Commodity', 'Market', 'Date', 'Modal_Price', 'Temperature', 'PRECTOTCORR']].head())
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return
    
    print("\nStep 2: Model Training")
    print("-" * 80)
    
    # Initialize model
    try:
        # Get unique commodities
        commodities = processed_df['Commodity'].unique()
        print(f"Found {len(commodities)} unique commodities in the dataset.")
        
        # Use LSTM model for time series forecasting
        print("Initializing LSTM model for price prediction...")
        model = CropPricePredictor(model_type='lstm', forecast_horizon=30, lookback_window=60)
        
        # Train models for top commodities by data volume
        commodity_counts = processed_df['Commodity'].value_counts()
        top_commodities = commodity_counts.index[:min(10, len(commodity_counts))]
        
        print(f"Training models for top {len(top_commodities)} commodities by data volume...")
        for commodity in top_commodities:
            print(f"\nTraining model for {commodity}")
            try:
                model.train_model(processed_df, commodity)
            except Exception as e:
                print(f"Error training model for {commodity}: {str(e)}")
                continue
        
        print("\nModel training complete.")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return
    
    print("\nStep 3: Crop Analysis and Recommendations")
    print("-" * 80)
    
    try:
        print("Analyzing seasonality and generating recommendations...")
        
        # Initialize the crop analyzer
        analyzer = CropAnalyzer(processed_df)
        
        # Check if Season column exists
        seasons = processed_df['Season'].unique() if 'Season' in processed_df.columns else [None]
        
        for season in seasons:
            season_name = season if season else "All Seasons"
            print(f"\nGenerating recommendations for {season_name}...")
            
            # Identify best crops
            ranked_crops, predictions = model.identify_best_crops(processed_df, prediction_horizon=90, season=season)
            
            if not ranked_crops:
                print(f"No suitable crops found for {season_name}.")
                continue
                
            # Visualize best crops
            print("Visualizing predictions for top crops...")
            top_crops = analyzer.visualize_best_crops(ranked_crops, predictions)
            
            # Create recommendation table
            recommendation_df = analyzer.create_crop_recommendation_table(ranked_crops, predictions, season)
            
            # Generate farmer report
            report = analyzer.generate_farmer_report(recommendation_df, top_crops, season)
            
            # Save outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "outputs"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Save recommendation dataframe
            recommendation_df.to_csv(f"{output_dir}/recommendations_{season_name.replace(' ', '_')}_{timestamp}.csv", 
                                  index=False)
            
            # Save report
            with open(f"{output_dir}/farmer_report_{season_name.replace(' ', '_')}_{timestamp}.txt", "w") as f:
                f.write(report)
                
            print(f"Saved recommendations and report for {season_name}.")
            
            # Perform seasonality analysis for top crops
            for crop in top_crops[:3]:  # Analyze top 3 crops
                print(f"\nAnalyzing seasonality for {crop}...")
                try:
                    # Seasonal patterns
                    analyzer.analyze_commodity_seasonality(crop)
                    plt.savefig(f"{output_dir}/{crop}_seasonality_{timestamp}.png")
                    
                    # Time series decomposition
                    analyzer.decompose_time_series(crop)
                    plt.savefig(f"{output_dir}/{crop}_decomposition_{timestamp}.png")
                    
                    # Weather correlation
                    analyzer.analyze_weather_price_correlation(crop)
                    plt.savefig(f"{output_dir}/{crop}_weather_correlation_{timestamp}.png")
                    
                except Exception as e:
                    print(f"Error analyzing {crop}: {str(e)}")
                    continue
            
        print("\nAnalysis complete. All outputs saved to the 'outputs' directory.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return
    
    print("\nStep 4: Model Evaluation")
    print("-" * 80)
    
    try:
        print("Evaluating model performance...")
        
        # Collect performance metrics across all models
        performance_metrics = {}
        for commodity, model_info in model.commodity_models.items():
            performance_metrics[commodity] = {
                'RMSE': model_info['rmse'],
                'MAPE': model_info['mape']
            }
        
        # Convert to dataframe for easy viewing
        metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
        
        # Save metrics
        metrics_df.to_csv(f"{output_dir}/model_performance_{timestamp}.csv")
        
        # Print summary statistics
        print("\nModel Performance Summary:")
        print(f"Average RMSE: {metrics_df['RMSE'].mean():.2f}")
        print(f"Average MAPE: {metrics_df['MAPE'].mean():.2f}%")
        print(f"Best performing model: {metrics_df['MAPE'].idxmin()} (MAPE: {metrics_df['MAPE'].min():.2f}%)")
        print(f"Worst performing model: {metrics_df['MAPE'].idxmax()} (MAPE: {metrics_df['MAPE'].max():.2f}%)")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    
    print("\nCrop Recommendation System execution completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
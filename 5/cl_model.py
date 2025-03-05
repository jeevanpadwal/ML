import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  # ✅ Correct import

from datetime import datetime
import joblib

# Load the data
def load_data(file_path):
    """
    Load and perform initial preprocessing of agricultural data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data preprocessing functions
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the agricultural data for modeling.
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Label encoding for categorical features (one encoder per column)
    categorical_columns = ['State Name', 'District Name', 'Market Name', 'Variety']
    
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))  # Ensure string type before encoding
    
    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Reported Date'], format='%d-%m-%Y', errors='coerce')
    
    # Extract more temporal features
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Day_of_week'] = data['Date'].dt.dayofweek
    
    # Ensure numerical columns are properly typed
    numeric_columns = ['Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 
                       'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)',
                       'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
                       'T2M', 'T2MDEW', 'PRECTOTCORR']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle missing values
    data = handle_missing_values(data, numeric_columns)
    
    # Create price range feature
    data['Price_Range'] = data['Max Price (Rs./Quintal)'] - data['Min Price (Rs./Quintal)']
    
    # Create lagged features for time series aspects
    data = create_lagged_features(data)
    
    # Drop unnecessary columns
    columns_to_drop = ['Reported Date', 'Date']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    print("Missing values after handling:", data.isnull().sum().sum())  

    return data


def handle_missing_values(df, numeric_columns):
    """
    Handle missing values in the dataset
    """
    # For categorical columns, fill with most frequent value
    categorical_columns = ['State Name', 'District Name', 'Market Name', 'Variety', 'Group']
    for col in categorical_columns:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # For numeric columns, use median for prices and arrivals, mean for weather
    for col in numeric_columns:
        if col in df.columns and df[col].isnull().any():
            if 'Price' in col or col == 'Arrivals (Tonnes)':
                # Group by region and crop type for more accurate imputation
                fill_values = df.groupby(['State Name', 'District Name', 'Group'])[col].transform('median')
                df[col] = df[col].fillna(fill_values)
                # If still missing, use overall median
                df[col] = df[col].fillna(df[col].median())
            else:
                # For weather data, use mean by region and date
                fill_values = df.groupby(['State Name', 'District Name', 'YEAR', 'DOY'])[col].transform('mean')
                df[col] = df[col].fillna(fill_values)
                # If still missing, use overall mean
                df[col] = df[col].fillna(df[col].mean())
    df = df.fillna(0)  # Replace remaining NaNs with 0 (or another sensible default)

    print("✅ Missing values after handling:", df.isnull().sum().sum())  # Verify it's 0
    
    return df

def create_lagged_features(df):
    """
    Create lagged features for time series prediction
    """
    # Group by location and crop
    groups = df.groupby(['State Name', 'District Name', 'Market Name', 'Group'])
    
    # Sort by date within each group
    df = df.sort_values(['State Name', 'District Name', 'Market Name', 'Group', 'YEAR', 'DOY'])
    
    # Create lagged price features
    for lag in [1, 7, 14, 30]:  # Previous day, week, 2 weeks, month
        df[f'Modal_Price_Lag_{lag}'] = groups['Modal Price (Rs./Quintal)'].shift(lag)
        df[f'Arrivals_Lag_{lag}'] = groups['Arrivals (Tonnes)'].shift(lag)
    
    # Create rolling averages
    for window in [7, 14, 30]:
        df[f'Modal_Price_Rolling_{window}'] = groups['Modal Price (Rs./Quintal)'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'Arrivals_Rolling_{window}'] = groups['Arrivals (Tonnes)'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
    
    # Drop rows with NaN values created by lagging
    df[lagged_cols] = df[lagged_cols].fillna(method='ffill').fillna(0)

    
    return df

# Feature engineering and selection
def prepare_features(df, target_col='Modal Price (Rs./Quintal)'):
    """
    Prepare features for model training
    """
    # Separate features and target
    y = df[target_col]
    
    # Drop the target column and other price columns to avoid data leakage
    price_cols = [col for col in df.columns if 'Price' in col and col != target_col]
    X = df.drop(columns=[target_col] + price_cols, errors='ignore')
    
    # Define categorical and numerical features
    categorical_features = ['State', 'District', 'Market', 'Variety', 'Group', 'Month', 'Week', 'Day_of_week']
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor

# Model training and evaluation
def train_models(X, y, preprocessor):
    """
    Train multiple models and select the best performing one
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name} Results: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")
        
        # Keep track of best model based on R²
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
    
    return best_model, results

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    # Extract the model from the pipeline
    estimator = model.named_steps['model']
    
    # Get feature importance if available
    if hasattr(estimator, 'feature_importances_'):
        # Get feature names from preprocessor
        preprocessor = model.named_steps['preprocessor']
        cat_features = preprocessor.transformers_[1][2]
        num_features = preprocessor.transformers_[0][2]
        
        # Get encoded feature names
        cat_encoder = preprocessor.transformers_[1][1]
        if hasattr(cat_encoder, 'get_feature_names_out'):
            encoded_cat_features = cat_encoder.get_feature_names_out(cat_features)
        else:
            encoded_cat_features = [f"{col}_{val}" for col in cat_features 
                                   for val in cat_encoder.categories_[cat_features.index(col)]]
        
        # Combine feature names
        all_features = list(num_features) + list(encoded_cat_features)
        
        # Sort feature importances
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        top_n = min(20, len(all_features))
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances')
        plt.barh(range(top_n), importances[indices][:top_n], align='center')
        plt.yticks(range(top_n), [all_features[i] for i in indices[:top_n]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    else:
        print("Model doesn't support feature importance analysis")
        return None

# Price prediction function
def predict_future_prices(model, df, days_ahead=30):
    """
    Predict prices for the next specified days
    """
    # Get the latest date in the dataset
    latest_year = df['YEAR'].max()
    latest_doy = df['DOY'].max()
    
    # Create a dataframe for predictions
    future_dfs = []
    
    # Group by location and crop
    for (state, district, market, group), group_df in df.groupby(['State Name', 'District Name', 'Market Name', 'Group']):
        if len(group_df) < 30:  # Skip groups with insufficient data
            continue
            
        # Get the latest data for this group
        latest_data = group_df.sort_values(['YEAR', 'DOY']).iloc[-1].copy()
        
        # Create prediction rows
        for i in range(1, days_ahead + 1):
            future_row = latest_data.copy()
            
            # Update date
            new_doy = latest_doy + i
            new_year = latest_year
            if new_doy > 365:
                new_doy -= 365
                new_year += 1
                
            future_row['YEAR'] = new_year
            future_row['DOY'] = new_doy
            
            # Convert DOY to month/week
            new_doy = int(new_doy)  # Convert float to int
            date = datetime(new_year, 1, 1) + pd.Timedelta(days=new_doy - 1)


            future_row['Month'] = date.month
            future_row['Week'] = date.isocalendar()[1]
            future_row['Day_of_week'] = date.weekday()
            
            # For lagged features, use the most recent actual and predicted values
            for lag in [1, 7, 14, 30]:
                # Initially use actual historical values
                if i <= lag:
                    idx = -lag + i - 1
                    future_row[f'Modal_Price_Lag_{lag}'] = group_df['Modal Price (Rs./Quintal)'].iloc[idx]
                    future_row[f'Arrivals_Lag_{lag}'] = group_df['Arrivals (Tonnes)'].iloc[idx]
                else:
                    # For longer predictions, use previously predicted values
                    lag_idx = i - lag - 1
                    future_row[f'Modal_Price_Lag_{lag}'] = future_dfs[-lag_idx]['Predicted_Price']
                    # Assume arrivals stay constant (can be improved)
                    future_row[f'Arrivals_Lag_{lag}'] = group_df['Arrivals (Tonnes)'].iloc[-1]
            
            # Use rolling averages based on historical data
            future_row[f'Modal_Price_Rolling_7'] = group_df['Modal Price (Rs./Quintal)'].iloc[-7:].mean()
            future_row[f'Modal_Price_Rolling_14'] = group_df['Modal Price (Rs./Quintal)'].iloc[-14:].mean()
            future_row[f'Modal_Price_Rolling_30'] = group_df['Modal Price (Rs./Quintal)'].iloc[-30:].mean()
            
            future_row[f'Arrivals_Rolling_7'] = group_df['Arrivals (Tonnes)'].iloc[-7:].mean()
            future_row[f'Arrivals_Rolling_14'] = group_df['Arrivals (Tonnes)'].iloc[-14:].mean()
            future_row[f'Arrivals_Rolling_30'] = group_df['Arrivals (Tonnes)'].iloc[-30:].mean()
            
            # Make prediction
            future_row_df = pd.DataFrame([future_row])
            X_future = future_row_df.drop(['Modal Price (Rs./Quintal)', 'Min Price (Rs./Quintal)', 
                                          'Max Price (Rs./Quintal)', 'Price_Range'], errors='ignore')
            
            predicted_price = model.predict(X_future)[0]
            
            # Store prediction
            future_row['Predicted_Price'] = predicted_price
            future_row['Prediction_Date'] = date
            
            future_dfs.append(future_row)
    
    if future_dfs:
        predictions_df = pd.DataFrame(future_dfs)
        return predictions_df
    else:
        return pd.DataFrame()

# Visualization functions
def plot_price_trends(df, crop_group=None, state=None):
    """
    Plot historical price trends with optional filtering
    """
    data = df.copy()
    
    # Convert date column if needed
    if 'Date' not in data.columns and 'Reported Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Reported Date'], format='%d-%m-%Y')
    
    # Filter data if requested
    if crop_group:
        data = data[data['Group'] == crop_group]
    if state:
        data = data[data['State Name'] == state]
    
    # Aggregate by date
    daily_prices = data.groupby('Date')['Modal Price (Rs./Quintal)'].mean().reset_index()
    
    # Plot trends
    plt.figure(figsize=(12, 6))
    plt.plot(daily_prices['Date'], daily_prices['Modal Price (Rs./Quintal)'])
    plt.title(f'Average Modal Price Trends {crop_group or "All Crops"} - {state or "All States"}')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs./Quintal)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_trends.png')

def plot_seasonal_patterns(df):
    """
    Plot seasonal patterns in prices
    """
    data = df.copy()
    
    # Convert date column if needed
    if 'Date' not in data.columns and 'Reported Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Reported Date'], format='%d-%m-%Y')
    
    # Extract month
    data['Month'] = data['Date'].dt.month
    
    # Calculate monthly averages
    monthly_prices = data.groupby(['Group', 'Month'])['Modal Price (Rs./Quintal)'].mean().reset_index()
    
    # Plot seasonal patterns for top crops
    top_crops = data['Group'].value_counts().head(5).index.tolist()
    
    plt.figure(figsize=(12, 8))
    
    for crop in top_crops:
        crop_data = monthly_prices[monthly_prices['Group'] == crop]
        plt.plot(crop_data['Month'], crop_data['Modal Price (Rs./Quintal)'], marker='o', label=crop)
    
    plt.title('Seasonal Price Patterns by Crop Group')
    plt.xlabel('Month')
    plt.ylabel('Average Price (Rs./Quintal)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('seasonal_patterns.png')

def plot_price_vs_arrivals(df):
    """
    Plot relationship between arrivals and prices
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Arrivals (Tonnes)', y='Modal Price (Rs./Quintal)', hue='Group', alpha=0.6)
    plt.title('Price vs. Arrivals Relationship')
    plt.xlabel('Arrivals (Tonnes)')
    plt.ylabel('Modal Price (Rs./Quintal)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_vs_arrivals.png')

def plot_weather_impact(df):
    """
    Visualize impact of weather parameters on crop prices
    """
    # Select weather columns
    weather_cols = ['T2M', 'PRECTOTCORR', 'CLOUD_AMT', 'WS50M']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(weather_cols):
        if col in df.columns:
            sns.scatterplot(data=df, x=col, y='Modal Price (Rs./Quintal)', ax=axes[i], alpha=0.5)
            axes[i].set_title(f'Price vs {col}')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weather_impact.png')

# Main function to run the entire analysis
def run_crop_price_analysis(file_path):
    """
    Run the complete crop price analysis pipeline
    """
    # Load and preprocess data
    print("Loading data...")
    df = load_data(file_path)
    if df is None:
        return
    
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Basic data exploration
    print("Generating visualizations...")
    plot_price_trends(df)
    plot_seasonal_patterns(df)
    plot_price_vs_arrivals(df)
    plot_weather_impact(df)
    
    # Prepare features for modeling
    print("Preparing features for modeling...")
    X, y, preprocessor = prepare_features(processed_df)
    
    # Train models
    print("Training models...")
    best_model, model_results = train_models(X, y, preprocessor)
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    feature_importance = analyze_feature_importance(best_model, X.columns)
    
    # Make future predictions
    print("Generating future price predictions...")
    predictions = predict_future_prices(best_model, processed_df)
    
    # Save model
    print("Saving model...")
    joblib.dump(best_model, 'crop_price_prediction_model.pkl')
    
    # Save predictions
    if not predictions.empty:
        predictions.to_csv('price_predictions.csv', index=False)
    
    print("Analysis complete!")
    
    return {
        'model': best_model,
        'results': model_results,
        'feature_importance': feature_importance,
        'predictions': predictions
    }

# Function to generate recommendations based on predictions
def generate_farmer_recommendations(predictions_df, df):
    """
    Generate actionable recommendations for farmers based on predictions
    """
    recommendations = []
    
    # Group predictions by crop and region
    for (state, district, market, group), group_preds in predictions_df.groupby(['State Name', 'District Name', 'Market Name', 'Group']):
        # Skip groups with too few predictions
        if len(group_preds) < 7:
            continue
            
        # Get historical data for this group
        historical = df[(df['State Name'] == state) & 
                        (df['District Name'] == district) & 
                        (df['Market Name'] == market) & 
                        (df['Group'] == group)]
        
        # Skip groups with too little historical data
        if len(historical) < 30:
            continue
            
        # Calculate current average price (last 7 days)
        current_price = historical['Modal Price (Rs./Quintal)'].iloc[-7:].mean()
        
        # Calculate predicted average price (next 7 days)
        next_week_price = group_preds['Predicted_Price'].iloc[:7].mean()
        
        # Calculate price change percentage
        price_change_pct = ((next_week_price - current_price) / current_price) * 100
        
        # Determine price trend
        if price_change_pct > 5:
            trend = "rising significantly"
            action = "consider delaying sales if storage is available"
        elif price_change_pct > 2:
            trend = "rising moderately"
            action = "monitor market closely before selling"
        elif price_change_pct < -5:
            trend = "falling significantly"
            action = "consider selling soon to avoid further price drops"
        elif price_change_pct < -2:
            trend = "falling moderately"
            action = "prepare for sale in the near term"
        else:
            trend = "stable"
            action = "sell based on your immediate needs"
        
        # Find best market in the region
        nearby_markets = df[(df['State'] == state) & 
                            (df['District'] == district) & 
                            (df['Group'] == group)]
        
        best_market = None
        best_price = 0
        
        if not nearby_markets.empty:
            market_avg_prices = nearby_markets.groupby('Market')['Modal Price (Rs./Quintal)'].mean()
            if not market_avg_prices.empty:
                best_market = market_avg_prices.idxmax()
                best_price = market_avg_prices.max()
        
        # Create recommendation
        recommendation = {
            'State': state,
            'District': district,
            'Market': market,
            'Crop': group,
            'Current_Price': current_price,
            'Predicted_Price_Next_Week': next_week_price,
            'Price_Change_Pct': price_change_pct,
            'Price_Trend': trend,
            'Recommended_Action': action,
            'Best_Market': best_market,
            'Best_Market_Price': best_price
        }
        
        recommendations.append(recommendation)
    
    return pd.DataFrame(recommendations)

# Usage guide function
def generate_usage_guide():
    """
    Generate a guide on how to use the prediction model
    """
    guide = """
    # Crop Price Prediction Model: User Guide for Farmers
    
    ## Overview
    This tool helps you predict crop prices based on historical trends, weather patterns, and market conditions. It can help you make better decisions about when to plant, harvest, and sell your crops.
    
    ## How to Use the Model
    
    ### Step 1: Install Required Software
    Make sure you have Python installed on your computer. This model requires Python 3.7 or higher.
    
    ### Step 2: Load Your Data
    The model needs historical data to make predictions. If you're using your own data, make sure it's in CSV format with the same column structure as the example data.
    
    ### Step 3: Run the Prediction
    ```python
    # Import the saved model
    import joblib
    model = joblib.load('crop_price_prediction_model.pkl')
    
    # Load and preprocess your data
    from crop_price_prediction import load_data, preprocess_data
    data = load_data('your_data.csv')
    processed_data = preprocess_data(data)
    
    # Generate predictions
    from crop_price_prediction import predict_future_prices
    predictions = predict_future_prices(model, processed_data, days_ahead=30)
    ```
    
    ### Step 4: Interpret the Results
    The predictions will show estimated prices for each crop and market combination for the next 30 days. Look for:
    - Upward trends: Consider delaying sales if storage is available
    - Downward trends: Consider selling sooner
    - Best markets: Check which nearby markets offer better prices
    
    ## Tips for Better Results
    1. Update your data regularly for more accurate predictions
    2. Pay attention to seasonal patterns in your crops
    3. Consider weather forecasts alongside price predictions
    4. Use the model as one tool in your decision-making process, not the only factor
    
    ## Getting Help
    If you need assistance with using this model, please contact your local agricultural extension office.
    """
    
    return guide

# If this script is run directly
if __name__ == "__main__":
    # This would be the entry point for running the analysis
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = pd.read_csv(file_path)

        print(df.head())

# Print column names
        print("Columns in dataset:", df.columns)
        results = run_crop_price_analysis(file_path)
        
        # Generate recommendations
        if results and 'predictions' in results and not results['predictions'].empty:
            recommendations = generate_farmer_recommendations(results['predictions'], load_data(file_path))
            recommendations.to_csv('farmer_recommendations.csv', index=False)
            
        # Generate usage guide
        with open('model_usage_guide.md', 'w') as f:
            f.write(generate_usage_guide())
    else:
        print("Please provide the path to the CSV file")
        print("Usage: python crop_price_prediction.py data.csv")
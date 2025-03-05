import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as XGBRegressor
import joblib
import argparse
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CropPricePredictionSystem:
    def __init__(self, data_path=None, prediction_days=30):
        """
        Initialize the Crop Price Prediction System
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the crop price data
        prediction_days : int
            Number of days to predict into the future
        """
        self.data_path = data_path
        self.data = None
        self.preprocessed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.prediction_days = prediction_days
        self.feature_importance = None
        self.predictions = None
        
    def load_data(self, data_path=None):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the crop price data
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            if data_path:
                self.data_path = data_path
            
            if not self.data_path:
                raise ValueError("Data path not provided")
                
            # Try to detect the CSV delimiter automatically
            with open(self.data_path, 'r') as f:
                first_line = f.readline()
                if ',' in first_line:
                    delimiter = ','
                elif ';' in first_line:
                    delimiter = ';'
                elif '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
            
            self.data = pd.read_csv(self.data_path, delimiter=delimiter)
            
            # Check if data was loaded successfully
            if self.data.empty:
                raise ValueError("No data was loaded from the file")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self):
        """
        Preprocess the loaded data by:
        - Handling missing values
        - Converting data types
        - Feature engineering
        - Encoding categorical variables
        
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Make a copy to avoid modifying the original data
            data = self.data.copy()
            
            # Convert date columns to datetime
            if 'Reported Date' in data.columns:
                data['Reported Date'] = pd.to_datetime(data['Reported Date'], 
                                                    format='%d-%m-%Y', errors='coerce')
                
                # Extract features from date
                data['Month'] = data['Reported Date'].dt.month
                data['Day'] = data['Reported Date'].dt.day
                data['DayOfWeek'] = data['Reported Date'].dt.dayofweek
            
            # Numeric columns
            numeric_cols = ['Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 
                            'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)',
                            'YEAR', 'DOY', 'WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 
                            'T2M', 'T2MDEW', 'PRECTOTCORR']
            
            # Ensure all numeric columns are converted properly
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN
            
            # ✅ FIX: Replace Inf and large values with NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # ✅ FIX: Fill NaN values with 0 for missing numeric data
            data[numeric_cols] = data[numeric_cols].fillna(0)

            # Create target variable
            self.target_column = 'Modal Price (Rs./Quintal)'
            
            # Feature engineering
            if 'Modal Price (Rs./Quintal)' in data.columns:
                data['Price_Lag_1'] = data.groupby(['State Name', 'District Name', 'Market Name', 'Variety'])['Modal Price (Rs./Quintal)'].shift(1)
                data['Price_Lag_7'] = data.groupby(['State Name', 'District Name', 'Market Name', 'Variety'])['Modal Price (Rs./Quintal)'].shift(7)
                data['Price_Rolling_Mean_7'] = data.groupby(['State Name', 'District Name', 'Market Name', 'Variety'])['Modal Price (Rs./Quintal)'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
                data['Price_Rolling_Mean_30'] = data.groupby(['State Name', 'District Name', 'Market Name', 'Variety'])['Modal Price (Rs./Quintal)'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
            
            if all(col in data.columns for col in ['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']):
                data['Price_Range'] = data['Max Price (Rs./Quintal)'] - data['Min Price (Rs./Quintal)']
                data['Price_Range_Pct'] = (data['Price_Range'] / (data['Modal Price (Rs./Quintal)'] + 0.01)) * 100  # Avoid division by zero
            
            if all(col in data.columns for col in ['Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']):
                data['Price_per_Arrival'] = data['Modal Price (Rs./Quintal)'] / (data['Arrivals (Tonnes)'] + 0.01)  # Avoid division by zero
            
            # Drop rows with NaN in target column
            data = data.dropna(subset=[self.target_column])
            
            # Separate features and target
            y = data[self.target_column]
            X = data.drop(self.target_column, axis=1)
            
            # Identify categorical and numeric columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if 'Reported Date' in numeric_cols:
                numeric_cols.remove('Reported Date')
            
            # Create preprocessing pipelines
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Median imputation
                ('scaler', StandardScaler())  # Standardization
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_cols),
                    ('num', numeric_transformer, numeric_cols)
                ],
                remainder='drop'  # Drop columns not specified
            )
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit and transform the training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Get feature names after one-hot encoding
            cat_features = []
            if categorical_cols:
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
            
            # Store preprocessed data
            self.X_train = X_train_transformed
            self.X_test = X_test_transformed
            self.y_train = y_train
            self.y_test = y_test
            self.preprocessor = preprocessor
            self.feature_names = cat_features + numeric_cols
            
            self.preprocessed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'preprocessor': preprocessor,
                'categorical_cols': categorical_cols,
                'numeric_cols': numeric_cols,
                'feature_names': self.feature_names,
                'last_date': data['Reported Date'].max() if 'Reported Date' in data.columns else None
            }
            
            print(f"Data preprocessing completed. Train set shape: {X_train_transformed.shape}")
            return self.preprocessed_data
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def train_models(self):
        """
        Train multiple regression models and select the best performing one
        
        Models:
        - Linear Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost
        
        Returns:
        --------
        dict
            Dictionary containing trained models
        """
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Preprocessed data not available. Call preprocess_data() first.")
            
            # Initialize models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=75),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': XGBRegressor.XGBRegressor(objective='reg:squarederror', random_state=42)
            }
            
            # Train and evaluate each model
            results = {}
            best_score = -float('inf')
            best_model = None
            best_model_name = None
            
            for name, model in models.items():
                print(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Evaluate the model
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                }
                
                print(f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                
                # Update best model if current one is better
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
            
            # Store the results
            self.models = results
            self.best_model = best_model
            self.best_model_name = best_model_name
            
            print(f"\nBest model: {best_model_name} with R² score of {best_score:.4f}")
            
            # Get feature importance for the best model
            if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                self.get_feature_importance()
            
            return results
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return None
    
    def tune_hyperparameters(self, model_name=None):
        """
        Perform hyperparameter tuning for the specified model
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model to tune. If None, tunes the best model.
            
        Returns:
        --------
        object
            Tuned model
        """
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Preprocessed data not available. Call preprocess_data() first.")
            
            if model_name is None:
                if self.best_model_name is None:
                    raise ValueError("No best model selected. Call train_models() first.")
                model_name = self.best_model_name
            
            print(f"Tuning hyperparameters for {model_name}...")
            
            if model_name == 'Random Forest':
                # Define the hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Initialize the model
                model = RandomForestRegressor(random_state=42)
                
            elif model_name == 'XGBoost':
                # Define the hyperparameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
                
                # Initialize the model
                model = XGBRegressor.XGBRegressor(objective='reg:squarederror', random_state=42)
                
            else:
                raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
            
            # Perform randomized search
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(self.X_train, self.y_train)
            
            # Get the best model
            best_params = search.best_params_
            best_score = search.best_score_
            tuned_model = search.best_estimator_
            
            print(f"Best parameters: {best_params}")
            print(f"Best cross-validation R² score: {best_score:.4f}")
            
            # Evaluate on test set
            y_pred = tuned_model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            print(f"Test set - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            # Update the best model if it's better than the current one
            if r2 > self.models[model_name]['r2']:
                self.models[model_name]['model'] = tuned_model
                self.models[model_name]['r2'] = r2
                self.models[model_name]['mae'] = mae
                self.models[model_name]['rmse'] = rmse
                
                if model_name == self.best_model_name:
                    self.best_model = tuned_model
                    print(f"Updated best model with tuned version")
                
                # Re-evaluate if this is now the best model
                if r2 > max(model_info['r2'] for model_info in self.models.values() if model_info['model'] is not tuned_model):
                    self.best_model = tuned_model
                    self.best_model_name = model_name
                    print(f"New best model: {model_name} with R² score of {r2:.4f}")
            
            return tuned_model
            
        except Exception as e:
            print(f"Error tuning hyperparameters: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """
        Get feature importance for the best model
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance scores
        """
        try:
            if self.best_model is None:
                raise ValueError("No best model selected. Call train_models() first.")
            
            if self.best_model_name not in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                print(f"Feature importance not available for {self.best_model_name}")
                return None
            
            # Get feature importance
            importance = self.best_model.feature_importances_
            
            # Map importance to feature names
            importance_df = pd.DataFrame({
                'Feature': self.feature_names[:len(importance)],
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
            
            self.feature_importance = importance_df
            
            print("Top 10 most important features:")
            print(importance_df.head(10))
            
            return importance_df
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
    
    def predict_future_prices(self, prediction_days=None):
        """
        Predict future prices for the specified number of days
        
        Parameters:
        -----------
        prediction_days : int, optional
            Number of days to predict into the future
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing predicted prices
        """
        try:
            if self.best_model is None:
                raise ValueError("No best model selected. Call train_models() first.")
            
            if prediction_days is not None:
                self.prediction_days = prediction_days
            
            # Get the last date from the data
            last_date = self.preprocessed_data['last_date']
            if last_date is None:
                print("Warning: No date information available, using sequential days")
                dates = [f"Day {i+1}" for i in range(self.prediction_days)]
            else:
                # Generate future dates
                dates = [last_date + timedelta(days=i+1) for i in range(self.prediction_days)]
            
            # For this example, we'll use the most recent row in the test set as a base for predictions
            # In a real system, you would need to update this with the latest data
            recent_data = self.preprocessed_data['X_test'].iloc[-1:].copy()
            
            # Prepare to store predictions
            predictions = []
            predicted_row = recent_data
            
            # Make predictions for each future day
            for i in range(self.prediction_days):
                # Transform the data
                X_pred = self.preprocessor.transform(predicted_row)
                
                # Make prediction
                price_pred = self.best_model.predict(X_pred)[0]
                
                # Store prediction
                predictions.append({
                    'Date': dates[i],
                    'Predicted Price': price_pred
                })
                
                # Update the row for the next prediction
                # In a real system, you'd update more fields
                if self.target_column in predicted_row.columns:
                    predicted_row[self.target_column] = price_pred
                
                # Update date-related features if available
                if isinstance(dates[i], datetime):
                    if 'Month' in predicted_row.columns:
                        predicted_row['Month'] = dates[i].month
                    if 'Day' in predicted_row.columns:
                        predicted_row['Day'] = dates[i].day
                    if 'DayOfWeek' in predicted_row.columns:
                        predicted_row['DayOfWeek'] = dates[i].weekday()
            
            # Convert to DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            self.predictions = predictions_df
            
            print(f"Future price predictions for {self.prediction_days} days:")
            print(predictions_df.head())
            
            return predictions_df
            
        except Exception as e:
            print(f"Error predicting future prices: {str(e)}")
            return None
    
    def visualize_data(self):
        """
        Generate visualizations for the data and predictions
        
        Returns:
        --------
        None
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # 1. Historical price trends
            if 'Reported Date' in self.data.columns and self.target_column in self.data.columns:
                data_with_date = self.data.dropna(subset=['Reported Date', self.target_column])
                if not data_with_date.empty:
                    axes[0, 0].plot(data_with_date['Reported Date'], data_with_date[self.target_column], 'b-')
                    axes[0, 0].set_title('Historical Price Trends')
                    axes[0, 0].set_xlabel('Date')
                    axes[0, 0].set_ylabel('Price (Rs./Quintal)')
                    axes[0, 0].grid(True)
                    axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Seasonal patterns (average price by month)
            if 'Reported Date' in self.data.columns and self.target_column in self.data.columns:
                self.data['Month'] = pd.to_datetime(self.data['Reported Date']).dt.month
                monthly_avg = self.data.groupby('Month')[self.target_column].mean().reset_index()
                if not monthly_avg.empty:
                    axes[0, 1].bar(monthly_avg['Month'], monthly_avg[self.target_column], color='green')
                    axes[0, 1].set_title('Average Price by Month')
                    axes[0, 1].set_xlabel('Month')
                    axes[0, 1].set_ylabel('Average Price (Rs./Quintal)')
                    axes[0, 1].set_xticks(range(1, 13))
                    axes[0, 1].grid(True, axis='y')
            
            # 3. Price vs. Arrivals
            if 'Arrivals (Tonnes)' in self.data.columns and self.target_column in self.data.columns:
                # Filter out extreme values for better visualization
                filtered_data = self.data[
                    (self.data['Arrivals (Tonnes)'] <= self.data['Arrivals (Tonnes)'].quantile(0.95)) &
                    (self.data[self.target_column] <= self.data[self.target_column].quantile(0.95))
                ]
                if not filtered_data.empty:
                    axes[1, 0].scatter(filtered_data['Arrivals (Tonnes)'], filtered_data[self.target_column], alpha=0.5)
                    axes[1, 0].set_title('Price vs. Arrivals')
                    axes[1, 0].set_xlabel('Arrivals (Tonnes)')
                    axes[1, 0].set_ylabel('Price (Rs./Quintal)')
                    axes[1, 0].grid(True)
            
            # 4. Price Prediction
            if self.predictions is not None:
                axes[1, 1].plot(range(len(self.predictions)), self.predictions['Predicted Price'], 'r-o')
                axes[1, 1].set_title(f'Predicted Prices for Next {len(self.predictions)} Days')
                axes[1, 1].set_xlabel('Days into the Future')
                axes[1, 1].set_ylabel('Predicted Price (Rs./Quintal)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save the figure
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            plt.savefig(os.path.join(output_dir, 'visualizations.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance visualization
            if self.feature_importance is not None:
                plt.figure(figsize=(12, 8))
                top_features = self.feature_importance.head(15)
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title(f'Top 15 Feature Importance for {self.best_model_name}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Visualizations saved to {output_dir} directory")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
    
    def generate_recommendations(self):
        """
        Generate actionable recommendations for farmers based on predicted prices
        
        Returns:
        --------
        list
            List of recommendations
        """
        try:
            if self.predictions is None:
                raise ValueError("No predictions available. Call predict_future_prices() first.")
            
            # Calculate statistics
            current_price = self.data[self.target_column].iloc[-1]
            predicted_prices = self.predictions['Predicted Price'].values
            avg_predicted_price = np.mean(predicted_prices)
            max_predicted_price = np.max(predicted_prices)
            min_predicted_price = np.min(predicted_prices)
            price_trend = np.polyfit(range(len(predicted_prices)), predicted_prices, 1)[0]
            
            # Initialize recommendations
            recommendations = []
            
            # Generate recommendations based on price trend
            if price_trend > 0:
                # Upward trend
                if max_predicted_price > current_price * 1.15:
                    recommendations.append(f"PRICE INCREASE EXPECTED: Prices are predicted to rise by approximately {((max_predicted_price - current_price) / current_price * 100):.1f}% in the next {self.prediction_days} days.")
                    
                    # When to sell
                    max_price_day = np.argmax(predicted_prices) + 1
                    recommendations.append(f"OPTIMAL SELLING TIME: Consider delaying sales until around day {max_price_day}, when prices are expected to peak at approximately Rs. {max_predicted_price:.2f} per quintal.")
                    
                    # Storage advice
                    if max_price_day > 7:
                        recommendations.append("STORAGE RECOMMENDATION: Ensure proper storage conditions to maintain crop quality while waiting for optimal selling time.")
                else:
                    recommendations.append(f"MODEST PRICE INCREASE: A slight upward trend in prices is predicted, with an average increase of approximately {((avg_predicted_price - current_price) / current_price * 100):.1f}% expected.")
                    recommendations.append("SELLING STRATEGY: Consider selling in batches over the next few weeks to average out price fluctuations.")
            
            elif price_trend < 0:
                # Downward trend
                if min_predicted_price < current_price * 0.85:
                    recommendations.append(f"PRICE DECREASE ALERT: Prices are predicted to fall by approximately {((current_price - min_predicted_price) / current_price * 100):.1f}% in the next {self.prediction_days} days.")
                    recommendations.append("IMMEDIATE ACTION: Consider selling your crop soon to avoid significant losses.")
                    
                    # Alternative markets
                    recommendations.append("MARKET DIVERSIFICATION: Explore alternative markets or value-added processing options that might offer better returns.")
                else:
                    recommendations.append(f"SLIGHT PRICE DECREASE: A modest downward trend in prices is predicted, with an average decrease of approximately {((current_price - avg_predicted_price) / current_price * 100):.1f}% expected.")
                    recommendations.append("BALANCED APPROACH: Consider selling part of your crop now and holding some for later in case the market stabilizes.")
            
            else:
                # Stable prices
                recommendations.append("STABLE MARKET: Prices are predicted to remain relatively stable over the next month.")
                recommendations.append("FLEXIBLE STRATEGY: Monitor the market closely and sell at your convenience, as significant price changes are not expected.")
            
            # Add crop-specific advice if we can identify the crop
            crops = self.data['Variety'].unique() if 'Variety' in self.data.columns else []
            if len(crops) == 1:
                crop = crops[0]
                recommendations.append(f"CROP-SPECIFIC ADVICE FOR {crop.upper()}: Based on current market conditions and historical patterns, consider exploring value-added processing or direct marketing to consumers to increase profitability.")
            
            # Add market-specific advice if we can identify the market
            markets = self.data['Market Name'].unique() if 'Market Name' in self.data.columns else []
            if len(markets) <= 3:
                market_list = ', '.join(markets)
                recommendations.append(f"MARKET-SPECIFIC INSIGHT: In {market_list}, historical data shows that price fluctuations are common. Consider exploring nearby markets for potentially better prices.")
            
            # Add seasonal advice based on the current month
            current_month = datetime.now().month
            if current_month in [11, 12, 1, 2]:  # Winter
                recommendations.append("SEASONAL FACTOR: Winter season typically affects storage and transportation. Ensure proper protection against cold to maintain crop quality.")
            elif current_month in [3, 4, 5]:  # Spring
                recommendations.append("SEASONAL FACTOR: Spring season often brings new crops to market. Be aware of increased competition and plan your selling strategy accordingly.")
            elif current_month in [6, 7, 8, 9]:  # Summer/Monsoon
                recommendations.append("SEASONAL FACTOR: Monsoon season can affect transportation and market access. Plan logistics carefully and consider weather forecasts when deciding when to transport your crop to market.")
            elif current_month in [10]:  # Autumn
                recommendations.append("SEASONAL FACTOR: Autumn harvest season often brings increased supply to markets. Consider early selling or storage options to avoid selling during peak supply periods.")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to an error."]
    
    def save_model(self, output_dir='models'):
        """
        Save the best model to disk
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the model
            
        Returns:
        --------
        str
            Path to the saved model
        """
        try:
            if self.best_model is None:
                raise ValueError("No best model available. Call train_models() first.")
            
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate a filename based on the model name and current date
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.best_model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            # Save the model
            joblib.dump(self.best_model, filepath)
            
            # Save the preprocessor for later use
            preprocessor_path = os.path.join(output_dir, f"preprocessor_{timestamp}.pkl")
            joblib.dump(self.preprocessor, preprocessor_path)
            
            print(f"Model saved to {filepath}")
            print(f"Preprocessor saved to {preprocessor_path}")
            
            # Also save the model evaluation metrics
            metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Model: {self.best_model_name}\n")
                f.write(f"R²: {self.models[self.best_model_name]['r2']:.4f}\n")
                f.write(f"MAE: {self.models[self.best_model_name]['mae']:.2f}\n")
                f.write(f"RMSE: {self.models[self.best_model_name]['rmse']:.2f}\n")
                if self.feature_importance is not None:
                    f.write("\nTop 10 Features:\n")
                    for i, row in self.feature_importance.head(10).iterrows():
                        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
            
            print(f"Metrics saved to {metrics_path}")
            
            return filepath
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, model_path, preprocessor_path=None):
        """
        Load a saved model from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        preprocessor_path : str, optional
            Path to the saved preprocessor
            
        Returns:
        --------
        object
            Loaded model
        """
        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Determine the model type
            if isinstance(model, LinearRegression):
                model_name = 'Linear Regression'
            elif isinstance(model, RandomForestRegressor):
                model_name = 'Random Forest'
            elif isinstance(model, GradientBoostingRegressor):
                model_name = 'Gradient Boosting'
            elif 'XGBRegressor' in str(type(model)):
                model_name = 'XGBoost'
            else:
                model_name = 'Unknown Model'
            
            # Update the model
            self.best_model = model
            self.best_model_name = model_name
            
            # Load the preprocessor if provided
            if preprocessor_path:
                self.preprocessor = joblib.load(preprocessor_path)
                print(f"Preprocessor loaded from {preprocessor_path}")
            
            print(f"Model '{model_name}' loaded from {model_path}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def generate_report(self, output_dir='reports'):
        """
        Generate a comprehensive report with all the analysis and predictions
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the report
            
        Returns:
        --------
        str
            Path to the saved report
        """
        try:
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate a filename based on the current date
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crop_price_prediction_report_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                # Title
                f.write("="*80 + "\n")
                f.write("CROP PRICE PREDICTION SYSTEM - ANALYSIS REPORT\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                # 1. Data Summary
                f.write("-"*80 + "\n")
                f.write("1. DATA SUMMARY\n")
                f.write("-"*80 + "\n")
                
                if self.data is not None:
                    f.write(f"Dataset Shape: {self.data.shape}\n")
                    f.write(f"Number of Records: {len(self.data)}\n")
                    
                    if 'Reported Date' in self.data.columns:
                        min_date = self.data['Reported Date'].min()
                        max_date = self.data['Reported Date'].max()
                        f.write(f"Date Range: {min_date} to {max_date}\n")
                    
                    if 'Variety' in self.data.columns:
                        crops = self.data['Variety'].unique()
                        f.write(f"Crops: {', '.join(crops[:5])}")
                        if len(crops) > 5:
                            f.write(f" and {len(crops)-5} more")
                        f.write("\n")
                    
                    if 'State Name' in self.data.columns:
                        states = self.data['State Name'].unique()
                        f.write(f"States: {', '.join(states[:5])}")
                        if len(states) > 5:
                            f.write(f" and {len(states)-5} more")
                        f.write("\n")
                    
                    if 'Market Name' in self.data.columns:
                        markets = self.data['Market Name'].unique()
                        f.write(f"Markets: {', '.join(markets[:5])}")
                        if len(markets) > 5:
                            f.write(f" and {len(markets)-5} more")
                        f.write("\n")
                    
                    if self.target_column in self.data.columns:
                        f.write(f"\nPrice Statistics:\n")
                        f.write(f"  Min Price: {self.data[self.target_column].min():.2f}\n")
                        f.write(f"  Max Price: {self.data[self.target_column].max():.2f}\n")
                        f.write(f"  Average Price: {self.data[self.target_column].mean():.2f}\n")
                        f.write(f"  Median Price: {self.data[self.target_column].median():.2f}\n")
                else:
                    f.write("No data available.\n")
                
                f.write("\n")
                
                # 2. Model Evaluation
                f.write("-"*80 + "\n")
                f.write("2. MODEL EVALUATION\n")
                f.write("-"*80 + "\n")
                
                if self.models:
                    f.write("Performance Metrics for Different Models:\n\n")
                    
                    # Table header
                    f.write(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10}\n")
                    f.write("-"*50 + "\n")
                    
                    # Table rows
                    for name, metrics in self.models.items():
                        f.write(f"{name:<20} {metrics['r2']:<10.4f} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f}\n")
                    
                    f.write("\n")
                    f.write(f"Best Model: {self.best_model_name} with R² score of {self.models[self.best_model_name]['r2']:.4f}\n")
                    
                    # Feature importance
                    if self.feature_importance is not None:
                        f.write("\nTop 10 Most Important Features:\n")
                        for i, row in self.feature_importance.head(10).iterrows():
                            f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
                else:
                    f.write("No model evaluation available.\n")
                
                f.write("\n")
                
                # 3. Price Predictions
                f.write("-"*80 + "\n")
                f.write("3. PRICE PREDICTIONS\n")
                f.write("-"*80 + "\n")
                
                if self.predictions is not None:
                    f.write(f"Predicted prices for the next {len(self.predictions)} days:\n\n")
                    
                    # Table header
                    f.write(f"{'Day':<10} {'Date':<12} {'Predicted Price':<15}\n")
                    f.write("-"*40 + "\n")
                    
                    # Table rows
                    for i, row in self.predictions.iterrows():
                        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], datetime) else str(row['Date'])
                        f.write(f"{i+1:<10} {date_str:<12} {row['Predicted Price']:<15.2f}\n")
                    
                    # Summary statistics
                    f.write("\nPrediction Summary:\n")
                    f.write(f"Average predicted price: {self.predictions['Predicted Price'].mean():.2f}\n")
                    f.write(f"Minimum predicted price: {self.predictions['Predicted Price'].min():.2f}\n")
                    f.write(f"Maximum predicted price: {self.predictions['Predicted Price'].max():.2f}\n")
                    
                    # Calculate the trend
                    price_trend = np.polyfit(range(len(self.predictions)), self.predictions['Predicted Price'].values, 1)[0]
                    trend_direction = "upward" if price_trend > 0 else "downward" if price_trend < 0 else "stable"
                    f.write(f"Overall trend: {trend_direction}\n")
                else:
                    f.write("No price predictions available.\n")
                
                f.write("\n")
                
                # 4. Recommendations
                f.write("-"*80 + "\n")
                f.write("4. RECOMMENDATIONS\n")
                f.write("-"*80 + "\n")
                
                recommendations = self.generate_recommendations()
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                else:
                    f.write("No recommendations available.\n")
            
            print(f"Report generated and saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None
    
    def run_full_analysis(self, data_path=None, prediction_days=None, output_dir='output'):
        """
        Run the full analysis pipeline from data loading to report generation
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the crop price data
        prediction_days : int, optional
            Number of days to predict into the future
        output_dir : str
            Directory to save outputs
            
        Returns:
        --------
        dict
            Dictionary containing paths to all outputs
        """
        try:
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 1. Load data
            if data_path:
                self.load_data(data_path)
            elif self.data_path:
                self.load_data()
            else:
                raise ValueError("No data path provided.")
            
            # 2. Preprocess data
            self.preprocess_data()
            
            # 3. Train models
            self.train_models()
            
            # 4. Tune the best model
            self.tune_hyperparameters()
            
            # 5. Get feature importance
            self.get_feature_importance()
            
            # 6. Predict future prices
            if prediction_days:
                self.predict_future_prices(prediction_days)
            else:
                self.predict_future_prices()
            
            # 7. Generate visualizations
            self.visualize_data()
            
            # 8. Save the model
            model_path = self.save_model(os.path.join(output_dir, 'models'))
            
            # 9. Generate report
            report_path = self.generate_report(os.path.join(output_dir, 'reports'))
            
            # Compile results
            result = {
                'model_path': model_path,
                'report_path': report_path,
                'visualizations_path': os.path.join(output_dir, 'visualizations.png'),
                'feature_importance_path': os.path.join(output_dir, 'feature_importance.png')
            }
            
            print("\nFull analysis completed successfully!")
            print("Results saved to:")
            for key, path in result.items():
                print(f"  - {key}: {path}")
            
            return result
            
        except Exception as e:
            print(f"Error running full analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Main function to run the Crop Price Prediction System from the command line
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Crop Price Prediction System')
    parser.add_argument('--data', '-d', type=str, help='Path to the CSV data file')
    parser.add_argument('--days', '-p', type=int, default=30, help='Number of days to predict (default: 30)')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--load-model', '-m', type=str, help='Path to a saved model to load')
    parser.add_argument('--load-preprocessor', '-pr', type=str, help='Path to a saved preprocessor to load')
    parser.add_argument('--tune', '-t', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--report-only', '-r', action='store_true', help='Generate report without retraining models')
    parser.add_argument('--visualize-only', '-v', action='store_true', help='Generate visualizations without retraining models')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = CropPricePredictionSystem(args.data, args.days)
    
    # Check if we're loading a saved model
    if args.load_model:
        # Load the data first
        if args.data:
            system.load_data()
            system.preprocess_data()
        
        # Load the model
        system.load_model(args.load_model, args.load_preprocessor)
        
        # Generate predictions and visualizations
        if args.data:
            system.predict_future_prices(args.days)
            system.visualize_data()
            system.generate_report(os.path.join(args.output, 'reports'))
            
        print("Model loaded and analysis completed.")
        return
    
    # Check if we're only generating visualizations
    if args.visualize_only:
        if args.data:
            system.load_data()
            system.preprocess_data()
            system.visualize_data()
            print("Visualizations generated.")
        else:
            print("Error: Data path is required for visualization.")
        return
    
    # Check if we're only generating a report
    if args.report_only:
        if args.data:
            system.load_data()
            system.preprocess_data()
            system.train_models()
            system.predict_future_prices(args.days)
            system.generate_report(os.path.join(args.output, 'reports'))
            print("Report generated.")
        else:
            print("Error: Data path is required for report generation.")
        return
    
    # Run the full analysis
    if args.data:
        # If tune flag is set, perform hyperparameter tuning
        if args.tune:
            system.load_data()
            system.preprocess_data()
            system.train_models()
            system.tune_hyperparameters()
            system.predict_future_prices(args.days)
            system.visualize_data()
            system.save_model(os.path.join(args.output, 'models'))
            system.generate_report(os.path.join(args.output, 'reports'))
            print("Analysis with hyperparameter tuning completed.")
        else:
            # Run the full analysis pipeline
            system.run_full_analysis(args.data, args.days, args.output)
    else:
        print("Error: Data path is required. Use --data option to specify the data file.")


if __name__ == "__main__":
    main()
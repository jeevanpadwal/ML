import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class CropPricePredictor:
    def __init__(self, model_type='lstm', forecast_horizon=30, lookback_window=60):
        """
        Initialize the price predictor
        Args:
            model_type: Type of model to use ('lstm', 'xgboost', 'prophet')
            forecast_horizon: Number of days to forecast
            lookback_window: Number of past days to use for prediction
        """
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = None
        self.commodity_models = {}

    def prepare_time_series_data(self, df, commodity, target_col='Modal_Price'):
        """
        Prepare time series data for the specified commodity
        Args:
            df: Processed dataframe
            commodity: The commodity to predict prices for
            target_col: The price column to predict
        Returns:
            X, y data for training
        """
        # Filter for the specific commodity
        commodity_df = df[df['Commodity'] == commodity].copy()
        
        # Sort by date
        commodity_df = commodity_df.sort_values('Date')
        
        if self.model_type in ['lstm', 'xgboost']:
            # Create sequences for time series prediction
            X, y = [], []
            
            # Select features to include
            weather_features = ['Temperature_scaled', 'Humidity_scaled', 'PRECTOTCORR_scaled', 
                               'PS_scaled', 'WS2M_scaled', 'Temperature_7day_avg']
            
            price_features = [f'{target_col}_scaled', f'{target_col}_7day_avg', f'{target_col}_30day_avg']
            
            # Add time features
            time_features = ['Month', 'Quarter', 'DayOfWeek']
            
            # Ensure all features exist in the dataframe
            features = [f for f in weather_features + price_features + time_features if f in commodity_df.columns]
            
            # Create sequences
            values = commodity_df[features + [target_col]].values
            for i in range(len(values) - self.lookback_window - self.forecast_horizon):
                X.append(values[i:i+self.lookback_window, :-1])  # All features except target
                y.append(values[i+self.lookback_window:i+self.lookback_window+self.forecast_horizon, -1])  # Target column
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y, features
            
        elif self.model_type == 'prophet':
            # For Prophet, we only need the date and target column
            prophet_df = commodity_df[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
            
            # Add regressor columns (weather features)
            regressor_cols = ['Temperature', 'Humidity', 'PRECTOTCORR']
            for col in regressor_cols:
                if col in commodity_df.columns:
                    prophet_df[col] = commodity_df[col].values
            
            return prophet_df, None, regressor_cols

    def build_lstm_model(self, input_shape, output_shape):
        """
        Build and compile LSTM model
        """
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_shape))
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_xgboost_model(self):
        """
        Build XGBoost model for time series
        """
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return model

    def build_prophet_model(self, regressors=None):
        """
        Build Prophet model with optional weather regressors
        """
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add weather regressors
        if regressors:
            for regressor in regressors:
                model.add_regressor(regressor)
                
        return model

    def train_model(self, df, commodity, target_col='Modal_Price'):
        """
        Train a model for the specified commodity
        """
        print(f"Training model for {commodity}...")
        
        if self.model_type in ['lstm', 'xgboost']:
            X, y, features = self.prepare_time_series_data(df, commodity, target_col)
            
            if len(X) < 10:  # Not enough data
                print(f"Not enough data for {commodity}. Skipping.")
                return None, None
            
            # Split data using time series split
            tscv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
            if self.model_type == 'lstm':
                # Reshape for LSTM [samples, time steps, features]
                input_shape = (X_train.shape[1], X_train.shape[2])
                output_shape = y_train.shape[1]
                
                model = self.build_lstm_model(input_shape, output_shape)
                
                # Train with early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
            elif self.model_type == 'xgboost':
                # Reshape for XGBoost [samples, features]
                X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                
                # XGBoost can only predict one step ahead, so we'll train separate models for each forecast day
                model = []
                for i in range(y_train.shape[1]):
                    xgb_model = self.build_xgboost_model()
                    xgb_model.fit(X_train_reshaped, y_train[:, i])
                    model.append(xgb_model)
            
            # Evaluate model
            if self.model_type == 'lstm':
                y_pred = model.predict(X_test)
            else:  # xgboost
                y_pred = np.zeros_like(y_test)
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                for i, m in enumerate(model):
                    y_pred[:, i] = m.predict(X_test_reshaped)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            self.commodity_models[commodity] = {
                'model': model,
                'features': features,
                'rmse': rmse,
                'mape': mape
            }
            
            return model, (rmse, mape)
            
        elif self.model_type == 'prophet':
            prophet_df, _, regressors = self.prepare_time_series_data(df, commodity, target_col)
            
            if len(prophet_df) < 30:  # Not enough data
                print(f"Not enough data for {commodity}. Skipping.")
                return None, None
            
            # Split data for evaluation
            train_size = int(len(prophet_df) * 0.8)
            train_df = prophet_df.iloc[:train_size]
            test_df = prophet_df.iloc[train_size:]
            
            # Build and train model
            model = self.build_prophet_model(regressors)
            model.fit(train_df)
            
            # Predict
            future = model.make_future_dataframe(periods=len(test_df), freq='D')
            # Add regressor values from test set
            for regressor in regressors:
                if regressor in test_df.columns:
                    future[regressor] = pd.concat([train_df[regressor], test_df[regressor]]).values
            
            forecast = model.predict(future)
            
            # Evaluate
            y_true = test_df['y'].values
            y_pred = forecast.iloc[-len(test_df):]['yhat'].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            self.commodity_models[commodity] = {
                'model': model,
                'rmse': rmse,
                'mape': mape
            }
            
            return model, (rmse, mape)

    def predict_future_prices(self, df, commodity, days_ahead=90):
        """
        Predict future prices for a commodity
        """
        if commodity not in self.commodity_models:
            print(f"No model found for {commodity}. Train a model first.")
            return None
        
        model_info = self.commodity_models[commodity]
        model = model_info['model']
        
        if self.model_type == 'lstm':
            # Get the latest data for prediction
            commodity_df = df[df['Commodity'] == commodity].copy().sort_values('Date')
            features = model_info['features']
            
            # Extract the most recent lookback window
            recent_data = commodity_df[features].values[-self.lookback_window:].reshape(1, self.lookback_window, len(features))
            
            # Predict for the number of days ahead
            # Since our model may be trained for forecast_horizon days, we'll predict recursively
            all_predictions = []
            current_window = recent_data.copy()
            
            for _ in range(0, days_ahead, self.forecast_horizon):
                next_predictions = model.predict(current_window)
                all_predictions.extend(next_predictions[0])
                
                # Update the window for the next prediction
                # This is a simplification; in practice, would need to update all features
                if days_ahead > self.forecast_horizon:
                    current_window = np.roll(current_window, -self.forecast_horizon, axis=1)
                    current_window[0, -self.forecast_horizon:, 0] = next_predictions[0]
            
            # Get the last date in the data
            last_date = commodity_df['Date'].max()
            
            # Create a dataframe with future dates and predictions
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
            predictions_df = pd.DataFrame({
                'Date': future_dates[:days_ahead],
                'Predicted_Price': all_predictions[:days_ahead]
            })
            
            return predictions_df
            
        elif self.model_type == 'xgboost':
            # Similar to LSTM but with reshaping for XGBoost
            commodity_df = df[df['Commodity'] == commodity].copy().sort_values('Date')
            features = model_info['features']
            
            recent_data = commodity_df[features].values[-self.lookback_window:].reshape(1, self.lookback_window * len(features))
            
            # Predict each step individually
            predictions = []
            for day in range(min(days_ahead, len(model))):
                xgb_model = model[day]
                pred = xgb_model.predict(recent_data)[0]
                predictions.append(pred)
            
            # For days beyond our model's forecast horizon, use the last prediction
            while len(predictions) < days_ahead:
                predictions.append(predictions[-1])
                
            # Create dataframe with predictions
            last_date = commodity_df['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': predictions
            })
            
            return predictions_df
            
        elif self.model_type == 'prophet':
            # Prophet makes future predictions directly
            future = model.make_future_dataframe(periods=days_ahead, freq='D')
            
            # Need to add regressors if used
            # This would require weather forecasts for future dates
            # For now, we'll use the last known values
            regressor_cols = ['Temperature', 'Humidity', 'PRECTOTCORR']
            commodity_df = df[df['Commodity'] == commodity].copy().sort_values('Date')
            
            for col in regressor_cols:
                if col in commodity_df.columns:
                    # Get the last 30 days average
                    last_30_avg = commodity_df[col].tail(30).mean()
                    # Set all future values to this average
                    # In practice, would use actual weather forecasts
                    future[col] = future[col].fillna(last_30_avg)
            
            # Make the forecast
            forecast = model.predict(future)
            
            # Extract just the future period
            predictions_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)
            predictions_df = predictions_df.rename(columns={
                'ds': 'Date', 
                'yhat': 'Predicted_Price',
                'yhat_lower': 'Lower_Bound',
                'yhat_upper': 'Upper_Bound'
            })
            
            return predictions_df

    def identify_best_crops(self, df, prediction_horizon=90, season=None):
        """
        Identify the best crops to grow based on predicted prices and season
        """
        all_predictions = {}
        crop_metrics = {}
        
        # Get unique commodities
        commodities = df['Commodity'].unique()
        
        # Filter by season if specified
        if season and 'Season' in df.columns:
            season_df = df[df['Season'] == season]
            if not season_df.empty:
                commodities = season_df['Commodity'].unique()
        
        # Predict prices for each commodity
        for commodity in commodities:
            if commodity in self.commodity_models:
                pred_df = self.predict_future_prices(df, commodity, days_ahead=prediction_horizon)
                
                if pred_df is not None:
                    all_predictions[commodity] = pred_df
                    
                    # Calculate metrics for ranking crops
                    avg_price = pred_df['Predicted_Price'].mean()
                    price_trend = pred_df['Predicted_Price'].iloc[-1] / pred_df['Predicted_Price'].iloc[0] - 1
                    
                    # Get model performance metrics
                    model_metrics = self.commodity_models[commodity]
                    prediction_accuracy = 100 - model_metrics['mape']  # Higher is better
                    
                    # Combine metrics for ranking
                    crop_metrics[commodity] = {
                        'avg_predicted_price': avg_price,
                        'price_trend_percent': price_trend * 100,
                        'prediction_accuracy': prediction_accuracy
                    }
        
        # Rank crops based on a combined score
        for commodity, metrics in crop_metrics.items():
            # Create a score combining price, trend and accuracy
            # Higher prices, positive trends, and accurate predictions are better
            metrics['score'] = (
                metrics['avg_predicted_price'] / 100 +  # Scale price 
                metrics['price_trend_percent'] * 0.5 +  # Weight trend
                metrics['prediction_accuracy'] * 0.2    # Weight accuracy
            )
        
        # Sort by score in descending order
        ranked_crops = sorted(crop_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return ranked_crops, all_predictions
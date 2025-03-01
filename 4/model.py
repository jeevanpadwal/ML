import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import os
from datetime import datetime

class CommodityPricePredictor:
    def __init__(self, sequence_length=10, commodity='Onion'):
        """
        Initialize the CommodityPricePredictor with configuration parameters.
        
        Args:
            sequence_length: Number of past days to use for prediction (default=10)
            commodity: Commodity to focus on (default='Onion')
        """
        self.sequence_length = sequence_length
        self.commodity = commodity
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the dataset.
        
        Args:
            filepath: Path to the dataset CSV file
            
        Returns:
            Preprocessed DataFrame with the selected commodity
        """
        print(f"Loading and preprocessing data from {filepath}...")
        
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Convert Arrival_Date to datetime format
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'],dayfirst=True)
        
        # Sort the data in ascending order of date
        df = df.sort_values('Arrival_Date')
        
        # Filter the dataset for the selected commodity
        commodity_df = df[df['Commodity'] == self.commodity].copy()
        
        if commodity_df.empty:
            raise ValueError(f"No data found for commodity: {self.commodity}")
            
        print(f"Found {len(commodity_df)} records for {self.commodity}")
        
        # Handle missing values in Modal_Price using forward fill
        if commodity_df['Modal_Price'].isnull().sum() > 0:
            print(f"Handling {commodity_df['Modal_Price'].isnull().sum()} missing values...")
            commodity_df['Modal_Price'] = commodity_df['Modal_Price'].fillna(method='ffill')
            
            # Drop any remaining rows with NaN values
            commodity_df = commodity_df.dropna(subset=['Modal_Price'])
            
        # Check if we still have enough data
        if len(commodity_df) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points after preprocessing. Need at least {self.sequence_length + 1}, but got {len(commodity_df)}")
            
        return commodity_df
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM training.
        
        Args:
            data: Normalized price data
            
        Returns:
            X: Input sequences, shape (n_samples, sequence_length, 1)
            y: Target values, shape (n_samples, 1)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        
        # Dense layers to refine predictions
        model.add(Dense(25))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, filepath, epochs=50, batch_size=16, validation_split=0.2):
        """
        Train the LSTM model on the given dataset.
        
        Args:
            filepath: Path to the dataset CSV file
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of data to use for validation
            
        Returns:
            Training history
        """
        # Load and preprocess data
        commodity_df = self.load_and_preprocess_data(filepath)
        
        # Extract and normalize the Modal_Price
        price_data = commodity_df['Modal_Price'].values.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(price_data)
        
        # Create sequences for LSTM training
        X, y = self.create_sequences(normalized_data)
        
        # Reshape X to match LSTM input requirements: [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Determine the split index
        split_idx = int(len(X) * (1 - validation_split))
        
        # Split the data into training and validation sets
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        # Build the model
        self.model = self.build_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'best_model_{self.commodity}.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]
        
        # Train the model
        print(f"Training model for {epochs} epochs with batch size {batch_size}...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best model
        self.model = load_model(f'best_model_{self.commodity}.h5')
        
        print("Model training completed.")
        return self.history
    
    def evaluate(self, filepath, test_split=0.1):
        """
        Evaluate the trained model on test data.
        
        Args:
            filepath: Path to the dataset CSV file
            test_split: Proportion of data to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Load and preprocess data
        commodity_df = self.load_and_preprocess_data(filepath)
        
        # Extract and normalize the Modal_Price
        price_data = commodity_df['Modal_Price'].values.reshape(-1, 1)
        normalized_data = self.scaler.transform(price_data)  # Use the fitted scaler
        
        # Create sequences
        X, y = self.create_sequences(normalized_data)
        
        # Reshape X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Get the test set (last test_split portion of the data)
        test_size = int(len(X) * test_split)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        # Make predictions
        y_pred_normalized = self.model.predict(X_test)
        
        # Inverse transform to get the actual prices
        y_test_actual = self.scaler.inverse_transform(y_test)
        y_pred_actual = self.scaler.inverse_transform(y_pred_normalized)
        
        # Calculate metrics
        rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual) * 100
        
        print(f"Evaluation Results for {self.commodity}:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual Prices', color='blue')
        plt.plot(y_pred_actual, label='Predicted Prices', color='red')
        plt.title(f'{self.commodity} Price Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.commodity}_price_prediction.png')
        plt.close()
        
        # Plot loss curves
        if self.history is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.history.history['loss'], label='Training Loss', color='blue')
            plt.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
            plt.title(f'{self.commodity} Model Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.commodity}_loss_curves.png')
            plt.close()
        
        return {
            'rmse': rmse,
            'mape': mape,
            'actual_prices': y_test_actual.flatten(),
            'predicted_prices': y_pred_actual.flatten()
        }
    
    def predict_future_prices(self, days_to_predict=30, filepath=None):
        """
        Predict future prices for the specified number of days.
        
        Args:
            days_to_predict: Number of days to predict into the future
            filepath: Path to the dataset CSV file (needed if model not trained yet)
            
        Returns:
            DataFrame with predicted prices
        """
        if self.model is None and filepath is not None:
            print("Model not trained yet. Training now...")
            self.train(filepath)
        elif self.model is None:
            raise ValueError("Model not trained and no filepath provided.")
        
        # Load the most recent data to start predictions from
        commodity_df = self.load_and_preprocess_data(filepath)
        price_data = commodity_df['Modal_Price'].values.reshape(-1, 1)
        normalized_data = self.scaler.transform(price_data)
        
        # Get the last sequence
        last_sequence = normalized_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Predict future prices
        future_prices = []
        current_sequence = last_sequence.copy()
        
        last_date = commodity_df['Arrival_Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
        
        for _ in range(days_to_predict):
            # Predict the next price
            next_price_normalized = self.model.predict(current_sequence)
            
            # Convert to actual price
            next_price = self.scaler.inverse_transform(next_price_normalized)[0, 0]
            future_prices.append(next_price)
            
            # Update the sequence for the next prediction
            next_price_normalized = next_price_normalized.reshape(1, 1, 1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_price_normalized, axis=1)
        
        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })
        
        # Plot the predictions
        plt.figure(figsize=(14, 7))
        
        # Plot historical prices
        historical_dates = commodity_df['Arrival_Date'][-30:].values  # Last 30 days
        historical_prices = commodity_df['Modal_Price'][-30:].values
        
        plt.plot(historical_dates, historical_prices, label='Historical Prices', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Price'], label='Predicted Prices', color='red')
        
        plt.title(f'{self.commodity} Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.commodity}_price_forecast.png')
        plt.close()
        
        return predictions_df
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_saved_model(self, filepath):
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage function
def run_commodity_price_prediction(data_filepath, commodity='Onion', sequence_length=10, 
                                  epochs=50, batch_size=16, predict_days=30):
    """
    Run the full commodity price prediction pipeline.
    
    Args:
        data_filepath: Path to the dataset CSV file
        commodity: Commodity to predict prices for
        sequence_length: Number of past days to use for prediction
        epochs: Number of training epochs
        batch_size: Batch size for training
        predict_days: Number of days to predict into the future
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    # Create the predictor
    predictor = CommodityPricePredictor(sequence_length=sequence_length, commodity=commodity)
    
    # Train the model
    predictor.train(data_filepath, epochs=epochs, batch_size=batch_size)
    
    # Evaluate the model
    evaluation = predictor.evaluate(data_filepath)
    
    # Predict future prices
    future_predictions = predictor.predict_future_prices(days_to_predict=predict_days, filepath=data_filepath)
    
    # Save the model
    model_filepath = f'{commodity}_price_predictor_{datetime.now().strftime("%Y%m%d")}.h5'
    predictor.save_model(model_filepath)
    
    return {
        'evaluation': evaluation,
        'predictions': future_predictions,
        'model_filepath': model_filepath
    }

# Example of running the pipeline
if __name__ == "__main__":
    # Replace with your dataset path
    data_path = "D:/programming/Python/ML/4/onion_pune.csv"
    
    # Run for Onion
    print("Running price prediction for Onion...")
    onion_results = run_commodity_price_prediction(
        data_path, 
        commodity='Onion',
        sequence_length=10,
        epochs=50,
        batch_size=16,
        predict_days=30
    )
    
    # Example of how to run for a different commodity
   
    
    print("Price prediction completed for multiple commodities.")
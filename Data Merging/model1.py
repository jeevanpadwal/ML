import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
data = pd.read_csv("D:/programming/Python/ML/Data Merging/merged_dataset.csv")  # Update with the actual dataset path

# Ensure correct date format
data["DATE"] = pd.to_datetime(data["DATE"])  

# Ensure data is sorted by date
data.sort_values("DATE", inplace=True)  
data.reset_index(drop=True, inplace=True)

# Select relevant features for price prediction
features = ["T2M", "CLOUD_AMT", "WS50M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
target = "Modal_Price"

# Create feature engineering
data['month'] = data['DATE'].dt.month
data['day_of_year'] = data['DATE'].dt.dayofyear

# Add these engineered features to the feature list
features.extend(['month', 'day_of_year'])

# Normalize data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

data[features] = feature_scaler.fit_transform(data[features])
# Reshape for sklearn compatibility
data[[target]] = target_scaler.fit_transform(data[[target]])

# Prepare time-series data for LSTM
sequence_length = 15  # Increased from 10 to 15 for more context
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[features].iloc[i:i+sequence_length].values)
    y.append(data[target].iloc[i+sequence_length])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create a more dense and complex LSTM model
def create_model():
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(sequence_length, len(features)))),
        Dropout(0.3),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        LSTM(80, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Use Adam optimizer with a smaller learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

# Build the model
model = create_model()
model.summary()

# Set up callbacks for fine-tuning
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('best_price_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train model with callbacks
epochs = 100  # Increased from 50
batch_size = 32  # Increased from 16
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

# Load the best model
model = load_model('best_price_model.h5')

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Mean Absolute Error: {mae}")

# Predict future prices
predictions = model.predict(X_test)

# Convert predictions back to original scale
predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions_original - y_test_original) ** 2))
print(f"Root Mean Square Error: {rmse}")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Plot actual vs predicted prices
test_dates = data['DATE'].iloc[sequence_length + len(X_train):sequence_length + len(X_train) + len(X_test)]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_original, label='Actual Prices')
plt.plot(test_dates, predictions_original, label='Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Save the model
model.save("enhanced_price_prediction_model.h5")
print("Enhanced model saved as enhanced_price_prediction_model.h5")

# Generate one year prediction
def predict_next_year(model, last_sequence, feature_scaler, target_scaler, data):
    """
    Predict prices for the next year based on the last available sequence
    """
    # Get the last date in the dataset
    last_date = data['DATE'].iloc[-1]
    
    # Create a list of dates for the next year
    future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]
    
    # Initialize prediction input with the last sequence
    pred_input = last_sequence
    
    # Store predictions
    predictions = []
    
    for future_date in future_dates:
        # Make prediction for the next day
        next_pred = model.predict(np.array([pred_input]))
        predictions.append(next_pred[0][0])
        
        # Create feature vector for the next day
        # Extract weather features from historical averages or use seasonal patterns
        # For this example, we'll use a simple approach of using the last day's features
        next_features = list(pred_input[-1])
        
        # Update month and day_of_year for the future date
        next_features[-2] = feature_scaler.transform([[future_date.month]])[0][0]  # month
        next_features[-1] = feature_scaler.transform([[future_date.dayofyear]])[0][0]  # day_of_year
        
        # Update the input sequence by removing the first item and adding the new prediction
        pred_input = np.vstack([pred_input[1:], next_features])
    
    # Convert predictions back to original scale
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return future_dates, predictions

# Get the last sequence from the dataset
last_sequence = data[features].iloc[-sequence_length:].values

# Predict prices for the next year
future_dates, future_predictions = predict_next_year(model, last_sequence, feature_scaler, target_scaler, data)

# Plot one year prediction
plt.figure(figsize=(15, 7))
plt.plot(future_dates, future_predictions, color='red', label='Predicted Future Prices')

# Add the last 60 days of actual data for context
last_60_days = data['DATE'].iloc[-60:]
last_60_days_prices = target_scaler.inverse_transform(data[[target]].iloc[-60:]).flatten()

plt.plot(last_60_days, last_60_days_prices, color='blue', label='Historical Prices')

plt.title('One Year Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add annotations for seasonal highs and lows
max_idx = np.argmax(future_predictions)
min_idx = np.argmin(future_predictions)
plt.annotate(f'Peak: {future_predictions[max_idx][0]:.2f}', 
             xy=(future_dates[max_idx], future_predictions[max_idx][0]),
             xytext=(10, 20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Low: {future_predictions[min_idx][0]:.2f}', 
             xy=(future_dates[min_idx], future_predictions[min_idx][0]),
             xytext=(10, -20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('one_year_forecast.png')
plt.show()

# Create a function to analyze the price pattern
def analyze_price_pattern(future_dates, future_predictions):
    """
    Analyze the predicted price pattern and identify trends
    """
    # Convert to dataframe for easier analysis
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions.flatten()
    })
    
    # Add month for seasonal analysis
    forecast_df['Month'] = forecast_df['Date'].dt.month
    forecast_df['Season'] = forecast_df['Month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall')
    
    # Calculate monthly averages
    monthly_avg = forecast_df.groupby('Month')['Predicted_Price'].mean().reset_index()
    monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: pd.to_datetime(f"2024-{x}-1").strftime('%b'))
    
    # Calculate seasonal averages
    seasonal_avg = forecast_df.groupby('Season')['Predicted_Price'].mean().reset_index()
    
    # Identify trend (upward, downward, or stable)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(forecast_df)), forecast_df['Predicted_Price'])
    
    trend = "Upward" if slope > 0.01 else "Downward" if slope < -0.01 else "Stable"
    
    # Create trend analysis plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Monthly averages
    plt.subplot(2, 2, 1)
    sns.barplot(x='Month_Name', y='Predicted_Price', data=monthly_avg, palette='viridis')
    plt.title('Average Monthly Prices')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    
    # Plot 2: Seasonal averages
    plt.subplot(2, 2, 2)
    sns.barplot(x='Season', y='Predicted_Price', data=seasonal_avg, 
                order=['Winter', 'Spring', 'Summer', 'Fall'], palette='viridis')
    plt.title('Average Seasonal Prices')
    plt.ylabel('Price')
    
    # Plot 3: Overall trend
    plt.subplot(2, 1, 2)
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'], color='blue', alpha=0.7)
    plt.plot([forecast_df['Date'].iloc[0], forecast_df['Date'].iloc[-1]], 
             [intercept, intercept + slope * len(forecast_df)], 'r--', 
             label=f'{trend} Trend (slope={slope:.4f})')
    plt.title(f'Price Trend Analysis: {trend}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_pattern_analysis.png')
    plt.show()
    
    # Return summary
    return {
        'trend': trend,
        'trend_slope': slope,
        'highest_month': monthly_avg.loc[monthly_avg['Predicted_Price'].idxmax(), 'Month_Name'],
        'lowest_month': monthly_avg.loc[monthly_avg['Predicted_Price'].idxmin(), 'Month_Name'],
        'best_season': seasonal_avg.loc[seasonal_avg['Predicted_Price'].idxmax(), 'Season'],
        'price_volatility': forecast_df['Predicted_Price'].std()
    }

# Analyze the predicted price pattern
pattern_analysis = analyze_price_pattern(future_dates, future_predictions)
print("\nPrice Pattern Analysis:")
for key, value in pattern_analysis.items():
    print(f"{key}: {value}")

# Let's also explore a function to recommend the best time to buy/sell
def generate_trading_recommendations(forecast_df):
    """
    Generate trading recommendations based on the price forecast
    """
    # Convert to DataFrame if it's not already
    if not isinstance(forecast_df, pd.DataFrame):
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions.flatten()
        })
    
    # Calculate moving averages
    forecast_df['MA7'] = forecast_df['Predicted_Price'].rolling(window=7).mean()
    forecast_df['MA30'] = forecast_df['Predicted_Price'].rolling(window=30).mean()
    
    # Calculate price momentum
    forecast_df['Price_Change'] = forecast_df['Predicted_Price'].diff()
    forecast_df['Price_Momentum'] = forecast_df['Price_Change'].rolling(window=7).mean()
    
    # Identify potential buy/sell points based on moving average crossover
    forecast_df['Signal'] = 0
    forecast_df.loc[forecast_df['MA7'] > forecast_df['MA30'], 'Signal'] = 1  # Buy signal
    forecast_df.loc[forecast_df['MA7'] < forecast_df['MA30'], 'Signal'] = -1  # Sell signal
    
    # Generate buy signals when signal changes from -1 to 1
    buy_signals = forecast_df[(forecast_df['Signal'].shift(1) == -1) & (forecast_df['Signal'] == 1)]
    
    # Generate sell signals when signal changes from 1 to -1
    sell_signals = forecast_df[(forecast_df['Signal'].shift(1) == 1) & (forecast_df['Signal'] == -1)]
    
    # Plot trading signals
    plt.figure(figsize=(15, 7))
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Price'], label='Predicted Price')
    plt.plot(forecast_df['Date'], forecast_df['MA7'], label='7-Day MA', alpha=0.7)
    plt.plot(forecast_df['Date'], forecast_df['MA30'], label='30-Day MA', alpha=0.7)
    
    # Mark buy and sell points
    plt.scatter(buy_signals['Date'], buy_signals['Predicted_Price'], 
                marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals['Date'], sell_signals['Predicted_Price'], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('Trading Signals Based on Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('trading_signals.png')
    plt.show()
    
    return {
        'buy_opportunities': buy_signals[['Date', 'Predicted_Price']].head(5).to_dict('records'),
        'sell_opportunities': sell_signals[['Date', 'Predicted_Price']].head(5).to_dict('records'),
        'best_buy_date': buy_signals.iloc[buy_signals['Predicted_Price'].idxmin()]['Date'] if not buy_signals.empty else None,
        'best_sell_date': sell_signals.iloc[sell_signals['Predicted_Price'].idxmax()]['Date'] if not sell_signals.empty else None
    }

# Generate trading recommendations
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions.flatten()
})
trading_recommendations = generate_trading_recommendations(forecast_df)

print("\nTrading Recommendations:")
print(f"Best time to buy: {trading_recommendations['best_buy_date']}")
print(f"Best time to sell: {trading_recommendations['best_sell_date']}")
print("\nTop Buy Opportunities:")
for opportunity in trading_recommendations['buy_opportunities'][:3]:
    print(f"Date: {opportunity['Date'].strftime('%Y-%m-%d')}, Predicted Price: {opportunity['Predicted_Price']:.2f}")
print("\nTop Sell Opportunities:")
for opportunity in trading_recommendations['sell_opportunities'][:3]:
    print(f"Date: {opportunity['Date'].strftime('%Y-%m-%d')}, Predicted Price: {opportunity['Predicted_Price']:.2f}")

print("\nModel and analysis complete. Check the saved images for detailed visualizations of predictions and patterns.")
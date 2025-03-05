import pandas as pd
import numpy as np
import re
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

class OnionCropPlanningModel:
    def __init__(self, csv_path):
        """
        Initialize the onion crop planning and price prediction model
        """
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None
        self.preprocessor = None
        self.model = None
        
    def preprocess_data(self):
        """
        Advanced data preprocessing with robust numeric conversion and outlier removal
        """
        # Read the CSV file
        self.df = pd.read_csv(self.csv_path)
        
        # Function to clean numeric columns
        def clean_numeric(value):
            if pd.isna(value):
                return np.nan
            # Remove commas and convert to float
            try:
                return float(str(value).replace(',', '').replace('Rs.', '').strip())
            except ValueError:
                return np.nan
        
        # Columns to clean
        numeric_columns = [
            'Arrivals (Tonnes)', 
            'Min Price (Rs./Quintal)', 
            'Max Price (Rs./Quintal)', 
            'Modal Price (Rs./Quintal)'
        ]
        
        # Clean numeric columns
        for col in numeric_columns:
            self.df[col] = self.df[col].apply(clean_numeric)
        
        # Remove outliers where Modal Price > 8000
        self.df = self.df[self.df['Modal Price (Rs./Quintal)'] <= 8000]
        
        # Convert Reported Date to datetime
        self.df['Reported Date'] = pd.to_datetime(self.df['Reported Date'], format='%d-%b-%y')
        
        # Extract additional time-based features
        self.df['Year'] = self.df['Reported Date'].dt.year
        self.df['Month'] = self.df['Reported Date'].dt.month
        self.df['Day'] = self.df['Reported Date'].dt.day
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['State Name', 'District Name', 'Market Name', 'Variety', 'Group']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col].astype(str))
            label_encoders[col] = le
        
        # Prepare features and target
        feature_columns = [
            'Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 
            'Max Price (Rs./Quintal)', 'Year', 'Month', 'Day',
            'State Name_Encoded', 'District Name_Encoded', 
            'Market Name_Encoded', 'Variety_Encoded', 'Group_Encoded'
        ]
        
        # Drop rows with NaN in critical columns
        X = self.df[feature_columns].dropna()
        y = self.df.loc[X.index, 'Modal Price (Rs./Quintal)']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocessing pipeline
        numeric_features = [
            'Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 
            'Max Price (Rs./Quintal)', 'Year', 'Month', 'Day'
        ]
        categorical_features = [
            'State Name_Encoded', 'District Name_Encoded', 
            'Market Name_Encoded', 'Variety_Encoded', 'Group_Encoded'
        ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
            ])
        
        # Fit preprocessor
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor
        }

        
    def create_deep_neural_network(self, input_shape):
        """
        Create an advanced deep neural network for price prediction
        """
        model = Sequential([
            Input(shape=(input_shape,)),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu', kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer, 
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_and_evaluate_model(self):
        """
        Train the model and generate comprehensive performance metrics
        """
        try:
            # Preprocess data
            data = self.preprocess_data()
            
            # Create and train the model
            model = self.create_deep_neural_network(data['X_train'].shape[1])
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=25, 
                restore_best_weights=True,
                min_delta=0.0001
            )
            
            lr_reducer = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.7, 
                patience=15, 
                min_lr=0.00001,
                verbose=1
            )
            
            # Train the model
            history = model.fit(
                data['X_train'], 
                data['y_train'], 
                epochs=300, 
                batch_size=32, 
                validation_split=0.2, 
                callbacks=[early_stopping, lr_reducer],
                verbose=1
            )
            
            # Predictions
            y_pred = model.predict(data['X_test']).flatten()
            
            # Calculate metrics
            metrics = {
                'Mean Squared Error': mean_squared_error(data['y_test'], y_pred),
                'Root Mean Squared Error': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'Mean Absolute Error': mean_absolute_error(data['y_test'], y_pred),
                'Mean Absolute Percentage Error': mean_absolute_percentage_error(data['y_test'], y_pred),
                'R-squared': r2_score(data['y_test'], y_pred)
            }
            
            # Crop Planning Insights
            crop_insights = self.generate_crop_planning_insights()
            
            return {
                'price_prediction_metrics': metrics,
                'crop_planning_insights': crop_insights
            }
        
        except Exception as e:
            print(f"Error in train_and_evaluate_model: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def generate_crop_planning_insights(self):
        """
        Generate insights for crop planning
        """
        # Analyze historical data for crop planning
        insights = {}
        
        # Average prices by variety
        variety_prices = self.df.groupby('Variety')['Modal Price (Rs./Quintal)'].agg(['mean', 'min', 'max'])
        insights['variety_price_summary'] = variety_prices.to_dict()
        
        # Best performing markets
        market_performance = self.df.groupby('Market Name').agg({
            'Modal Price (Rs./Quintal)': ['mean', 'max'],
            'Arrivals (Tonnes)': 'sum'
        })
        insights['top_markets'] = market_performance.sort_values(('Arrivals (Tonnes)', 'sum'), ascending=False).head(5).to_dict()
        
        # Seasonal price trends
        self.df['Season'] = pd.cut(
            self.df['Reported Date'].dt.month, 
            bins=[0, 3, 6, 9, 12], 
            labels=['Winter', 'Spring', 'Summer', 'Autumn']
        )
        seasonal_prices = self.df.groupby('Season')['Modal Price (Rs./Quintal)'].mean()
        insights['seasonal_price_trends'] = seasonal_prices.to_dict()
        
        return insights

def main(csv_path):
    """
    Main function to execute the onion crop planning model
    """
    try:
        # Initialize and run model
        model = OnionCropPlanningModel(csv_path)
        results = model.train_and_evaluate_model()
        
        # Print results
        print("\n--- Price Prediction Metrics ---")
        for metric, value in results['price_prediction_metrics'].items():
            print(f"{metric}: {value}")
        
        print("\n--- Crop Planning Insights ---")
        print("Variety Price Summary:", results['crop_planning_insights']['variety_price_summary'])
        print("\nTop Markets:", results['crop_planning_insights']['top_markets'])
        print("\nSeasonal Price Trends:", results['crop_planning_insights']['seasonal_price_trends'])
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

# Sample usage
if __name__ == "__main__":
    csv_file_path = r"C:/Users/ADMIN/Downloads/onion_pune_quanity_market_price.csv"
    results = main(csv_file_path)
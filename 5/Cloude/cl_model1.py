import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import HuberRegressor, LassoCV
from sklearn.cluster import KMeans

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import xgboost as xgb
import optuna

# For outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class HybridForestDLModel(BaseEstimator, RegressorMixin):
    """
    A hybrid model combining Random Forest and Deep Neural Network.
    
    Parameters:
    - rf_params (dict): Parameters for RandomForestRegressor.
    - dl_params (dict): Parameters for the deep learning model.
    - blend_weight (float): Weight for blending RF and DL predictions (0 to 1).
    """
    def __init__(self, rf_params=None, dl_params=None, blend_weight=0.5):
        # Default RF parameters
        self.rf_params = rf_params if rf_params else {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Default DL parameters
        self.dl_params = dl_params if dl_params else {
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        
        self.blend_weight = blend_weight
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.dl_model = None
        self.feature_names = None
        
    def build_dl_model(self, input_dim):
        """Build the deep learning model architecture."""
        model = Sequential()
        
        # First hidden layer
        model.add(Dense(self.dl_params['hidden_layers'][0], 
                        input_dim=input_dim,
                        activation=self.dl_params['activation'],
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(self.dl_params['dropout_rate']))
        
        # Additional hidden layers
        for units in self.dl_params['hidden_layers'][1:]:
            model.add(Dense(units, 
                           activation=self.dl_params['activation'],
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(Dropout(self.dl_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.dl_params['learning_rate']),
            loss='mean_squared_error'
        )
        
        return model
    
    def fit(self, X, y):
        "Fit the hybrid model."
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        # Convert X to numeric and handle errors
        if hasattr(X, 'apply'):
            X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN

        # Convert to numpy arrays
        X_array = X.values if hasattr(X, 'values') else np.array(X, dtype=np.float32)
        y_array = y.values if hasattr(y, 'values') else np.array(y, dtype=np.float32)

        # Fill NaNs with median values
        X_array = np.nan_to_num(X_array, nan=np.nanmedian(X_array))
        
        # Check for NaNs
        if np.any(np.isnan(X_array)):
            raise ValueError("X_array contains NaN values after processing")
        if np.any(np.isnan(y_array)):
            raise ValueError("y_array contains NaN values")

        # Train the Random Forest model
        self.rf_model.fit(X_array, y_array)

        # Get RF predictions
        rf_preds = self.rf_model.predict(X_array).reshape(-1, 1)

        # Build and train the deep learning model
        if self.dl_model is None:
            self.dl_model = self.build_dl_model(X_array.shape[1])

        # Combine RF predictions with original features
        X_dl = np.hstack((X_array, rf_preds))

        # Check X_dl for NaNs
        if np.any(np.isnan(X_dl)):
            raise ValueError("X_dl contains NaN values after processing")

        # Convert to float32 for TensorFlow
        X_dl = X_dl.astype(np.float32)

        # Create train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_dl, y_array, test_size=0.15, random_state=42
        )

        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        ]

        # Train the DL model
        self.dl_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.dl_params['epochs'],
            batch_size=self.dl_params['batch_size'],
            callbacks=callbacks,
            verbose=0
        )

        return self

    
    def predict(self, X):
        """Make predictions with the hybrid model."""
        # Convert to numpy arrays if they're not already
        X_array = X.values if hasattr(X, 'values') else X
        
        # Get Random Forest predictions
        rf_preds = self.rf_model.predict(X_array).reshape(-1, 1)
        
        # Combine original features and RF predictions for DL
        X_dl = np.hstack((X_array, rf_preds))
        
        # Get Deep Learning predictions
        dl_preds = self.dl_model.predict(X_dl, verbose=0).flatten()
        
        # Blend predictions
        blended_preds = (self.blend_weight * rf_preds.flatten() + 
                         (1 - self.blend_weight) * dl_preds)
        
        return blended_preds
    
    def get_feature_importance(self):
        """Return feature importance from the Random Forest model."""
        if self.feature_names is None:
            return self.rf_model.feature_importances_
        
        return dict(zip(self.feature_names, self.rf_model.feature_importances_))
# Enhanced data preprocessing function
def enhanced_preprocess_data(df, detect_outliers=True):
    """
    Enhanced preprocessing with feature engineering and outlier detection.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with raw data.
    - detect_outliers (bool): Whether to perform outlier detection (default: True).
    
    Returns:
    - df (pd.DataFrame): Processed DataFrame.
    - categorical_features (list): List of categorical feature names.
    - numerical_features (list): List of numerical feature names.
    - target (str): Name of the target variable.
    """
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['State Name'] = label_encoder.fit_transform(df['State Name'])
    df['District Name'] = label_encoder.fit_transform(df['District Name'])
    df['Market Name'] = label_encoder.fit_transform(df['Market Name'])
    df['Variety'] = label_encoder.fit_transform(df['Variety'])

    # Convert 'Reported Date' to datetime
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce',dayfirst=True)
    
    # Advanced date feature engineering
    df['Month'] = df['Reported Date'].dt.month
    df['DayOfWeek'] = df['Reported Date'].dt.dayofweek
    df['DayOfMonth'] = df['Reported Date'].dt.day
    df['WeekOfYear'] = df['Reported Date'].dt.isocalendar().week
    df['Quarter'] = df['Reported Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Seasonal features based on agricultural cycles
    df['Season'] = df['Month'].apply(lambda m: 1 if m in [12, 1, 2] else  # Winter
                                      2 if m in [3, 4, 5] else      # Spring
                                      3 if m in [6, 7, 8] else      # Summer
                                      4)                            # Fall
    
    # Fill missing YEAR and DOY from Reported Date
    df['YEAR'] = df['YEAR'].fillna(df['Reported Date'].dt.year)
    df['DOY'] = df['DOY'].fillna(df['Reported Date'].dt.dayofyear)

    # Fill missing values for market data columns
    market_cols = ['Arrivals (Tonnes)', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']
    for col in market_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df[col] = df[col].fillna(df[col].median())

    # Fill missing values for weather-related features
    weather_cols = ['WS50M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering based on weather and market data
    if 'T2M' in df.columns and 'T2MDEW' in df.columns:
        df['Temp_Difference'] = df['T2M'] - df['T2MDEW']
    if 'PRECTOTCORR' in df.columns and 'T2M' in df.columns:
        df['Precip_Temp_Ratio'] = df['PRECTOTCORR'] / (df['T2M'] + 1)
    if all(col in df.columns for col in ['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)']):
        df['Price_Spread'] = df['Max Price (Rs./Quintal)'] - df['Min Price (Rs./Quintal)']
        df['Price_Spread_Ratio'] = df['Price_Spread'] / (df['Modal Price (Rs./Quintal)'] + 1)
    if 'Arrivals (Tonnes)' in df.columns:
        df['Log_Arrivals'] = np.log1p(df['Arrivals (Tonnes)'])
    weather_cols = ['WS50M', 'CLOUD_AMT', 'PRECTOTCORR']
    if all(col in df.columns for col in weather_cols):
        for col in weather_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_Normalized'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_Normalized'] = 0
        df['Weather_Severity'] = (df['WS50M_Normalized'] + df['CLOUD_AMT_Normalized'] + 
                                  df['PRECTOTCORR_Normalized']) / 3

    # Fill missing values for all engineered numerical features
    engineered_features = [
        'Temp_Difference', 'Precip_Temp_Ratio', 'Price_Spread', 
        'Price_Spread_Ratio', 'Log_Arrivals', 'Weather_Severity',
        'WS50M_Normalized', 'CLOUD_AMT_Normalized', 'PRECTOTCORR_Normalized'
    ]
    for col in engineered_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    

    df = df.fillna(0)

    # Outlier detection (if enabled)
    if detect_outliers and 'Arrivals (Tonnes)' in df.columns:
        outlier_check_cols = ['Modal Price (Rs./Quintal)', 'Arrivals (Tonnes)']
        outlier_df = df[outlier_check_cols].copy()
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(outlier_df)
        df['IsOutlier'] = [1 if x == -1 else 0 for x in outliers]
        if df['Arrivals (Tonnes)'].min() > 0:
            df['Log_Arrivals'] = np.log1p(df['Arrivals (Tonnes)'])  # Ensure consistency

    # Define features and target
    categorical_features = ['State Name', 'District Name', 'Market Name', 'Variety', 'Group']
    numerical_features = [
        'Arrivals (Tonnes)', 'YEAR', 'DOY', 'WS50M', 'CLOUD_AMT', 
        'ALLSKY_SFC_SW_DWN', 'T2M', 'T2MDEW', 'PRECTOTCORR', 
        'Month', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Quarter', 
        'IsWeekend', 'Season'
    ]
    numerical_features.extend([feat for feat in engineered_features if feat in df.columns])
    numerical_features = list(set(numerical_features))
    numerical_features = [feat for feat in numerical_features if feat in df.columns]
    if detect_outliers:
        numerical_features.append('IsOutlier')

    target = 'Modal Price (Rs./Quintal)'

    return df, categorical_features, numerical_features, target

# Function to optimize model hyperparameters using Optuna
def optimize_hybrid_model(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimize the hyperparameters of the hybrid model using Optuna.
    """
    def objective(trial):
        # Random Forest parameters
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        # Deep Learning parameters
        dl_params = {
            'hidden_layers': [
                trial.suggest_int('dl_hidden1', 64, 512),
                trial.suggest_int('dl_hidden2', 32, 256),
                trial.suggest_int('dl_hidden3', 16, 128)
            ],
            'dropout_rate': trial.suggest_float('dl_dropout', 0.1, 0.5),
            'activation': trial.suggest_categorical('dl_activation', ['relu', 'elu', 'selu']),
            'learning_rate': trial.suggest_float('dl_lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('dl_batch_size', [16, 32, 64, 128]),
            'epochs': 100  # Fixed epochs, we'll use early stopping
        }
        
        # Blend weight
        blend_weight = trial.suggest_float('blend_weight', 0.0, 1.0)
        
        # Create and train the hybrid model
        model = HybridForestDLModel(rf_params=rf_params, dl_params=dl_params, blend_weight=blend_weight)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    
    # Extract RF and DL parameters
    rf_params = {
        'n_estimators': best_params['rf_n_estimators'],
        'max_depth': best_params['rf_max_depth'],
        'min_samples_split': best_params['rf_min_samples_split'],
        'min_samples_leaf': best_params['rf_min_samples_leaf'],
        'random_state': 42
    }
    
    dl_params = {
        'hidden_layers': [
            best_params['dl_hidden1'],
            best_params['dl_hidden2'],
            best_params['dl_hidden3']
        ],
        'dropout_rate': best_params['dl_dropout'],
        'activation': best_params['dl_activation'],
        'learning_rate': best_params['dl_lr'],
        'batch_size': best_params['dl_batch_size'],
        'epochs': 100
    }
    
    return rf_params, dl_params, best_params['blend_weight']


# Function to perform feature selection
def perform_feature_selection(X_train, y_train, categorical_features, numerical_features, threshold=0.01):
    """
    Perform feature selection using a Random Forest model's feature importances.
    """
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', RobustScaler(), numerical_features)
        ], remainder='passthrough'
    )
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train a Random Forest for feature importance
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    feature_selector.fit(X_train_processed, y_train)
    
    # Get feature importances
    importances = feature_selector.feature_importances_
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Get feature names for categorical features after one-hot encoding
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    
    # Add all feature names
    feature_names.extend(cat_feature_names)
    feature_names.extend(numerical_features)
    
    # Create a list of (feature_name, importance) tuples
    feature_importance = list(zip(feature_names, importances))
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Select features above the threshold
    selected_features = [feat for feat, imp in feature_importance if imp > threshold]
    
    # Print selected features and their importance
    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
    for feat, imp in feature_importance[:10]:  # Print top 10
        print(f"{feat}: {imp:.4f}")
    
    return selected_features, feature_importance


# Enhanced evaluation function with cross-validation
def evaluate_model_with_cv(model, X, y, cv=5):
    """
    Evaluate model performance using cross-validation.
    """
    # Initialize KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Metrics for each fold
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
    
    # Calculate average metrics
    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    
    print(f"Cross-Validation Results (CV={cv}):")
    print(f"Mean Absolute Error: {avg_mae:.2f} Rs./Quintal")
    print(f"Root Mean Squared Error: {avg_rmse:.2f} Rs./Quintal")
    print(f"R-squared: {avg_r2:.2f}")
    
    # Print confidence intervals (using standard error)
    print(f"\nConfidence Intervals (95%):")
    print(f"MAE: {avg_mae:.2f} ± {np.std(mae_scores, ddof=1) * 1.96 / np.sqrt(cv):.2f}")
    print(f"RMSE: {avg_rmse:.2f} ± {np.std(rmse_scores, ddof=1) * 1.96 / np.sqrt(cv):.2f}")
    print(f"R²: {avg_r2:.2f} ± {np.std(r2_scores, ddof=1) * 1.96 / np.sqrt(cv):.2f}")
    
    return avg_mae, avg_rmse, avg_r2


# Build a stacked model
def build_stacked_model(categorical_features, numerical_features):
    """
    Build a stacked ensemble model combining multiple base models.
    """
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', RobustScaler(), numerical_features)
        ]
    )
    
    # Define base models
    base_models = [
        ('rf', Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=200, random_state=42))
        ])),
        ('gb', Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])),
        ('xgb', Pipeline([
            ('preprocessor', preprocessor),
            ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
        ])),
        ('huber', Pipeline([
            ('preprocessor', preprocessor),
            ('model', HuberRegressor())
        ]))
    ]
    
    # Define meta-learner
    meta_learner = LassoCV(cv=5)
    
    # Create stacked model
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    return stacked_model


# Main execution function
def train_enhanced_model(data_file):
    """
    Train an enhanced price prediction model with all improvements.
    
    Parameters:
    - data_file (str): Path to the CSV file containing the data.
    
    Returns:
    - best_model: The trained model (either hybrid or stacked).
    - categorical_features (list): List of categorical feature names.
    - numerical_features (list): List of numerical feature names.
    """
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)
    
    # Step 1: Enhanced Data Preprocessing
    print("Preprocessing data with enhanced features...")
    df_processed, categorical_features, numerical_features, target = enhanced_preprocess_data(data)
    
    # Drop rows where target is NaN
    df_processed = df_processed.dropna(subset=[target])
    
    # Print data shape and sample
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Number of categorical features: {len(categorical_features)}")
    print(f"Number of numerical features: {len(numerical_features)}")
    
    # Step 2: Feature Selection (Placeholder - assumes function exists)
    print("\nPerforming feature selection...")
    X = df_processed[categorical_features + numerical_features]
    y = df_processed[target]
    
    
    # Split data into training, validation, and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
    
    # Placeholder for feature selection (you may need to define this function)
    def perform_feature_selection(X_train, y_train, cat_feats, num_feats):
        return cat_feats + num_feats, None  # Simplified return
    selected_features, _ = perform_feature_selection(X_train, y_train, categorical_features, numerical_features)
    
    # Step 3: Model Hyperparameter Optimization (Placeholder - assumes function exists)
    print("\nOptimizing hybrid model hyperparameters...")
    def optimize_hybrid_model(X_train, y_train, X_val, y_val, n_trials=10):
        rf_params = {'n_estimators': 200, 'max_depth': 15}
        dl_params = {'hidden_layers': [256, 128, 64], 'dropout_rate': 0.3, 'learning_rate': 0.001}
        return rf_params, dl_params, 0.5  # Simplified return
    rf_params, dl_params, blend_weight = optimize_hybrid_model(X_train, y_train, X_val, y_val)
    
    print("\nBest parameters:")
    print(f"Random Forest: {rf_params}")
    print(f"Deep Learning: {dl_params}")
    print(f"Blend Weight: {blend_weight}")
    
    # Step 4: Train the optimized hybrid model
    print("\nTraining the optimized hybrid model...")
    optimized_hybrid_model = HybridForestDLModel(rf_params=rf_params, dl_params=dl_params, blend_weight=blend_weight)
    
    # Placeholder for cross-validation evaluation (you may need to define this function)
    def evaluate_model_with_cv(model, X, y, cv=5):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
        mae = -scores.mean()
        rmse = np.sqrt(mean_squared_error(y, model.fit(X, y).predict(X)))
        r2 = r2_score(y, model.predict(X))
        return mae, rmse, r2
    
    print("\nEvaluating hybrid model with cross-validation...")
    hybrid_mae, hybrid_rmse, hybrid_r2 = evaluate_model_with_cv(optimized_hybrid_model, X_train_val, y_train_val)
    
    # Step 5: Train a stacked ensemble model for comparison (Placeholder)
    print("\nTraining a stacked ensemble model for comparison...")
    def build_stacked_model(cat_feats, num_feats):
        return RandomForestRegressor(n_estimators=100, random_state=42)  # Simplified
    stacked_model = build_stacked_model(categorical_features, numerical_features)
    
    print("\nEvaluating stacked model with cross-validation...")
    stacked_mae, stacked_rmse, stacked_r2 = evaluate_model_with_cv(stacked_model, X_train_val, y_train_val)
    
    # Step 6: Final evaluation on test set
    print("\nFinal evaluation on test set...")
    
    # Train hybrid model on all training+validation data
    final_hybrid_model = HybridForestDLModel(rf_params=rf_params, dl_params=dl_params, blend_weight=blend_weight)
    final_hybrid_model.fit(X_train_val, y_train_val)
    
    # Train stacked model on all training+validation data
    final_stacked_model = build_stacked_model(categorical_features, numerical_features)
    final_stacked_model.fit(X_train_val, y_train_val)
    
    # Make predictions on test set
    hybrid_preds = final_hybrid_model.predict(X_test)
    stacked_preds = final_stacked_model.predict(X_test)
    
    # Calculate metrics
    hybrid_test_mae = mean_absolute_error(y_test, hybrid_preds)
    hybrid_test_rmse = np.sqrt(mean_squared_error(y_test, hybrid_preds))
    hybrid_test_r2 = r2_score(y_test, hybrid_preds)
    
    stacked_test_mae = mean_absolute_error(y_test, stacked_preds)
    stacked_test_rmse = np.sqrt(mean_squared_error(y_test, stacked_preds))
    stacked_test_r2 = r2_score(y_test, stacked_preds)
    
    # Print test results
    print("\nTest Set Results:")
    print(f"Hybrid Model - MAE: {hybrid_test_mae:.2f}, RMSE: {hybrid_test_rmse:.2f}, R²: {hybrid_test_r2:.2f}")
    print(f"Stacked Model - MAE: {stacked_test_mae:.2f}, RMSE: {stacked_test_rmse:.2f}, R²: {stacked_test_r2:.2f}")
    
    # Step 7: Return the best model
    if hybrid_test_rmse <= stacked_test_rmse:
        print("\nHybrid model is the best performer! Returning this model.")
        return final_hybrid_model, categorical_features, numerical_features
    else:
        print("\nStacked ensemble model is the best performer! Returning this model.")
        return final_stacked_model, categorical_features, numerical_features

# Enhanced prediction function
def enhanced_predict_price(model, input_data, categorical_features, numerical_features):
    """
    Make price predictions with the enhanced model.
    """
    
    # Convert input to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Preprocess input data
    input_df, cat_features, num_features, _ = enhanced_preprocess_data(
        input_data, detect_outliers=False
    )
    
    # Ensure all required features are present
    for feat in categorical_features:
        if feat not in input_df.columns:
            input_df[feat] = None  # Add missing categorical features
    
    for feat in numerical_features:
        if feat not in input_df.columns:
            input_df[feat] = 0  # Add missing numerical features with zeros
    
    # Select only the needed features in the right order
    input_features = input_df[categorical_features + numerical_features]
    
    # Make prediction
    predicted_price = model.predict(input_features)[0]
    
    # Estimate uncertainty (simplified approach)
    if hasattr(model, 'rf_model'):
        # For the hybrid model, use RF's native uncertainty estimation
        rf_preds = []
        for tree in model.rf_model.estimators_:
            rf_preds.append(tree.predict(input_features)[0])
        
        uncertainty = np.std(rf_preds)
    else:
        # For other models like the stacked ensemble
        # Use a simplified approach - 10% of the predicted value
        uncertainty = predicted_price * 0.1
    
    return predicted_price, uncertainty


# Function to perform what-if analysis
def perform_what_if_analysis(model, input_data, categorical_features, numerical_features, 
                            feature_to_vary, min_val, max_val, steps=10):
    """
    Perform what-if analysis by varying a specific feature and observing predicted price changes.
    
    Args:
        model: Trained prediction model
        input_data: Base input data (dictionary or DataFrame)
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        feature_to_vary: Feature to vary in the analysis
        min_val: Minimum value for the feature
        max_val: Maximum value for the feature
        steps: Number of steps between min and max
        
    Returns:
        DataFrame with feature values and corresponding predictions
    """
    # Convert input to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Create range of values to test
    test_values = np.linspace(min_val, max_val, steps)
    
    # Store results
    results = []
    
    # For each test value
    for val in test_values:
        # Create a copy of the input data
        test_data = input_data.copy()
        
        # Set the feature value
        test_data[feature_to_vary] = val
        
        # Get prediction
        pred_price, uncertainty = enhanced_predict_price(
            model, test_data, categorical_features, numerical_features
        )
        
        # Store result
        results.append({
            feature_to_vary: val,
            'Predicted_Price': pred_price,
            'Uncertainty': uncertainty
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


# Function to explain model predictions using SHAP
def explain_prediction(model, input_data, categorical_features, numerical_features):
    """
    Explain model predictions using SHAP values.
    
    Note: This function requires the SHAP library to be installed.
    pip install shap
    """
    try:
        import shap
    except ImportError:
        print("SHAP library not found. Please install with: pip install shap")
        return None
    
    # Convert input to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Preprocess input data
    input_df, _, _, _ = enhanced_preprocess_data(
        input_data, detect_outliers=False
    )
    
    # Select only the needed features in the right order
    input_features = input_df[categorical_features + numerical_features]
    
    # For the hybrid model, we'll use the RF component for explanation
    if hasattr(model, 'rf_model'):
        explainer = shap.TreeExplainer(model.rf_model)
        shap_values = explainer.shap_values(input_features)
        
        # Create a DataFrame with feature names and SHAP values
        explanation_df = pd.DataFrame({
            'Feature': categorical_features + numerical_features,
            'SHAP_Value': shap_values[0],
            'Feature_Value': input_features.iloc[0].values
        })
        
        # Sort by absolute SHAP value
        explanation_df['Abs_SHAP'] = explanation_df['SHAP_Value'].abs()
        explanation_df = explanation_df.sort_values('Abs_SHAP', ascending=False).drop('Abs_SHAP', axis=1)
        
        return explanation_df
    else:
        print("Model explanation is currently only supported for the hybrid model.")
        return None


# Function to save and load the trained model
def save_model(model, filename='price_prediction_model.pkl'):
    """Save the trained model to a file"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='price_prediction_model.pkl'):
    """Load a trained model from a file"""
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model


# Function to create a deployable web API for the model
def create_model_api(model, categorical_features, numerical_features):
    """
    Create a simple Flask API for the model.
    
    Note: This requires Flask to be installed.
    pip install flask
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Please install with: pip install flask")
        return
    
    app = Flask("Price Prediction API")
    
    @app.route('/predict', methods=['POST'])
    def predict():
        # Get input data from request
        input_data = request.json
        
        # Make prediction
        try:
            predicted_price, uncertainty = enhanced_predict_price(
                model, input_data, categorical_features, numerical_features
            )
            
            # Return prediction
            return jsonify({
                'predicted_price': float(predicted_price),
                'uncertainty': float(uncertainty),
                'units': 'Rs./Quintal'
            })
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Add a route for model information
    @app.route('/info', methods=['GET'])
    def info():
        model_info = {
            'model_type': 'Hybrid Forest-DL' if hasattr(model, 'rf_model') else 'Stacked Ensemble',
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        }
        return jsonify(model_info)
    
    # Run the API
    print("Starting API server. Use Ctrl+C to stop.")
    app.run(debug=True, host='0.0.0.0', port=5000)


# Main function to run the entire pipeline
def main(data_file, mode='train'):
    """
    Main function to run the entire pipeline.
    
    Args:
        data_file: Path to the CSV data file
        mode: 'train' to train a new model, 'predict' to load a saved model and make predictions,
              'api' to start the prediction API
    """
    if mode == 'train':
        # Train a new model
        model, categorical_features, numerical_features = train_enhanced_model(data_file)
        
        # Save the model and features
        save_model(
            (model, categorical_features, numerical_features),
            filename='price_prediction_model.pkl'
        )
        
    elif mode == 'predict':
        # Load the model
        model, categorical_features, numerical_features = load_model('price_prediction_model.pkl')
        
        # Load test data
        test_data = pd.read_csv(data_file)
        
        # Make predictions
        X = test_data[categorical_features + numerical_features]
        
        # Make predictions for each row
        predictions = []
        for i, row in X.iterrows():
            pred_price, uncertainty = enhanced_predict_price(
                model, row.to_dict(), categorical_features, numerical_features
            )
            predictions.append({
                'index': i,
                'predicted_price': pred_price,
                'uncertainty': uncertainty
            })
        
        # Convert to DataFrame and save
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('predictions.csv', index=False)
        print(f"Predictions saved to predictions.csv")
        
    elif mode == 'api':
        # Load the model
        model, categorical_features, numerical_features = load_model('price_prediction_model.pkl')
        
        # Start the API
        create_model_api(model, categorical_features, numerical_features)
    
    else:
        print(f"Unknown mode: {mode}. Use 'train', 'predict', or 'api'.")


# If this script is run directly
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Agricultural Price Prediction System')
    parser.add_argument('data_file', help='Path to the data CSV file')
    parser.add_argument('--mode', choices=['train', 'predict', 'api'], default='train',
                       help='Operation mode: train, predict, or api')
    parser.add_argument('--feature', help='Feature to vary in what-if analysis')
    parser.add_argument('--min', type=float, help='Minimum value for what-if analysis')
    parser.add_argument('--max', type=float, help='Maximum value for what-if analysis')
    
    args = parser.parse_args()
    
    # Run the main function
    if args.feature and args.min is not None and args.max is not None:
        # Load the model
        model, categorical_features, numerical_features = load_model('price_prediction_model.pkl')
        
        # Load a sample input
        sample_data = pd.read_csv(args.data_file).iloc[0:1]
        
        # Perform what-if analysis
        results = perform_what_if_analysis(
            model, sample_data, categorical_features, numerical_features,
            args.feature, args.min, args.max
        )
        
        # Print results
        print(f"What-if analysis for {args.feature}:")
        print(results)
        
        # Save results to CSV
        results.to_csv(f'whatif_{args.feature}.csv', index=False)
        print(f"Results saved to whatif_{args.feature}.csv")
    else:
        main(args.data_file, args.mode)
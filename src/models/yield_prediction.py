
"""
📊 Yield Prediction Module - Annam AI
====================================

This module implements crop yield prediction using machine learning and time series analysis.
It combines weather data, soil information, and historical yield data to forecast agricultural production.

Key Features:
- Traditional ML models (Random Forest, XGBoost, LightGBM)
- Time series forecasting (ARIMA, LSTM)
- Feature engineering from weather and soil data
- Multi-variate prediction models
- Integration with weather APIs
- Model evaluation and visualization
- Export for deployment

Author: Annam AI Team
License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data processing
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep Learning for time series
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import logging

# Set up logging
logger = logging.getLogger('AnnamAI.YieldPrediction')


class WeatherDataCollector:
    """Collect weather data from free APIs for yield prediction."""

    def __init__(self, api_key=None):
        self.api_key = api_key  # OpenWeatherMap API key (free tier)
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def get_historical_weather(self, lat, lon, start_date, end_date):
        """
        Get historical weather data for a location.
        Note: This is a simplified example. In practice, you'd use APIs like:
        - OpenWeatherMap Historical
        - NASA Power API (free)
        - NOAA Climate Data
        """
        # Simulate weather data for demonstration
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Simulate realistic weather patterns
        np.random.seed(42)
        n_days = len(date_range)

        weather_data = pd.DataFrame({
            'date': date_range,
            'temperature_max': np.random.normal(25, 8, n_days),
            'temperature_min': np.random.normal(15, 6, n_days),
            'temperature_avg': np.random.normal(20, 7, n_days),
            'humidity': np.random.normal(65, 15, n_days),
            'precipitation': np.random.exponential(2, n_days),
            'wind_speed': np.random.exponential(3, n_days),
            'solar_radiation': np.random.normal(200, 50, n_days),
            'pressure': np.random.normal(1013, 10, n_days)
        })

        # Ensure realistic ranges
        weather_data['humidity'] = np.clip(weather_data['humidity'], 20, 100)
        weather_data['precipitation'] = np.clip(weather_data['precipitation'], 0, 50)
        weather_data['wind_speed'] = np.clip(weather_data['wind_speed'], 0, 30)
        weather_data['solar_radiation'] = np.clip(weather_data['solar_radiation'], 50, 400)

        return weather_data

    def get_nasa_power_data(self, lat, lon, start_date, end_date):
        """
        Get NASA POWER API data (free agricultural weather data).
        This is a real API that provides free agricultural weather data.
        """
        # NASA POWER API endpoint
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        parameters = [
            "T2M",      # Temperature at 2 meters
            "T2M_MAX",  # Maximum Temperature at 2 meters  
            "T2M_MIN",  # Minimum Temperature at 2 meters
            "RH2M",     # Relative Humidity at 2 meters
            "PRECTOTCORR",  # Precipitation
            "WS2M",     # Wind Speed at 2 meters
            "ALLSKY_SFC_SW_DWN"  # Solar irradiance
        ]

        url = f"{base_url}?parameters={','.join(parameters)}&community=AG&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()

                # Convert to DataFrame
                df_list = []
                for param, values in data['properties']['parameter'].items():
                    param_df = pd.DataFrame(list(values.items()), columns=['date', param])
                    param_df['date'] = pd.to_datetime(param_df['date'])
                    df_list.append(param_df.set_index('date'))

                weather_df = pd.concat(df_list, axis=1).reset_index()

                # Rename columns to match our schema
                weather_df = weather_df.rename(columns={
                    'T2M': 'temperature_avg',
                    'T2M_MAX': 'temperature_max',
                    'T2M_MIN': 'temperature_min',
                    'RH2M': 'humidity',
                    'PRECTOTCORR': 'precipitation',
                    'WS2M': 'wind_speed',
                    'ALLSKY_SFC_SW_DWN': 'solar_radiation'
                })

                logger.info(f"Successfully downloaded NASA POWER data for {len(weather_df)} days")
                return weather_df

        except Exception as e:
            logger.warning(f"Failed to get NASA POWER data: {e}")

        # Fallback to simulated data
        return self.get_historical_weather(lat, lon, start_date, end_date)


class YieldDataProcessor:
    """Process and prepare yield data for modeling."""

    def __init__(self):
        self.crop_coefficients = {
            'wheat': {'temp_opt': 20, 'precip_opt': 450, 'growing_days': 120},
            'corn': {'temp_opt': 25, 'precip_opt': 600, 'growing_days': 140},
            'rice': {'temp_opt': 28, 'precip_opt': 1200, 'growing_days': 120},
            'soybean': {'temp_opt': 22, 'precip_opt': 500, 'growing_days': 110},
            'cotton': {'temp_opt': 27, 'precip_opt': 700, 'growing_days': 180}
        }

    def create_sample_yield_data(self, n_samples=1000):
        """Create sample yield data for demonstration."""
        np.random.seed(42)

        crops = ['wheat', 'corn', 'rice', 'soybean', 'cotton']
        years = list(range(2000, 2024))
        regions = ['North', 'South', 'East', 'West', 'Central']

        data = []

        for _ in range(n_samples):
            crop = np.random.choice(crops)
            year = np.random.choice(years)
            region = np.random.choice(regions)

            # Soil parameters
            soil_ph = np.random.normal(6.5, 0.8)
            soil_organic_matter = np.random.normal(3.2, 1.0)
            soil_nitrogen = np.random.normal(45, 15)
            soil_phosphorus = np.random.normal(25, 8)
            soil_potassium = np.random.normal(180, 40)

            # Weather aggregates (growing season)
            temp_avg = np.random.normal(self.crop_coefficients[crop]['temp_opt'], 3)
            temp_max_avg = temp_avg + np.random.normal(8, 2)
            temp_min_avg = temp_avg - np.random.normal(8, 2)

            total_precipitation = np.random.normal(self.crop_coefficients[crop]['precip_opt'], 100)
            avg_humidity = np.random.normal(60, 12)
            avg_solar_radiation = np.random.normal(220, 30)

            # Growing degree days
            gdd = max(0, (temp_avg - 10) * self.crop_coefficients[crop]['growing_days'])

            # Yield calculation (simplified model)
            base_yield = {
                'wheat': 4.5, 'corn': 9.8, 'rice': 6.2, 
                'soybean': 3.1, 'cotton': 2.8
            }[crop]

            # Weather stress factors
            temp_stress = 1 - abs(temp_avg - self.crop_coefficients[crop]['temp_opt']) / 10
            precip_stress = 1 - abs(total_precipitation - self.crop_coefficients[crop]['precip_opt']) / 300

            # Soil fertility factor
            soil_factor = (
                (soil_ph - 5.5) / 3 * 0.3 +
                soil_organic_matter / 5 * 0.3 +
                soil_nitrogen / 60 * 0.2 +
                soil_phosphorus / 40 * 0.1 +
                soil_potassium / 200 * 0.1
            )

            # Calculate yield with some randomness
            yield_tons_per_hectare = (
                base_yield * 
                (0.7 + 0.3 * temp_stress) * 
                (0.7 + 0.3 * precip_stress) * 
                (0.8 + 0.2 * soil_factor) * 
                (0.9 + 0.1 * np.random.random())
            )

            data.append({
                'crop': crop,
                'year': year,
                'region': region,
                'soil_ph': soil_ph,
                'soil_organic_matter': soil_organic_matter,
                'soil_nitrogen': soil_nitrogen,
                'soil_phosphorus': soil_phosphorus,
                'soil_potassium': soil_potassium,
                'temperature_avg': temp_avg,
                'temperature_max_avg': temp_max_avg,
                'temperature_min_avg': temp_min_avg,
                'total_precipitation': total_precipitation,
                'avg_humidity': avg_humidity,
                'avg_solar_radiation': avg_solar_radiation,
                'growing_degree_days': gdd,
                'yield_tons_per_hectare': yield_tons_per_hectare
            })

        return pd.DataFrame(data)

    def engineer_features(self, df):
        """Engineer additional features for yield prediction."""
        df = df.copy()

        # Temperature-related features
        df['temp_range'] = df['temperature_max_avg'] - df['temperature_min_avg']
        df['heat_stress'] = (df['temperature_max_avg'] > 35).astype(int)
        df['cold_stress'] = (df['temperature_min_avg'] < 5).astype(int)

        # Precipitation features
        df['drought_stress'] = (df['total_precipitation'] < 200).astype(int)
        df['flood_stress'] = (df['total_precipitation'] > 800).astype(int)

        # Soil fertility index
        df['soil_fertility_index'] = (
            df['soil_organic_matter'] * 0.3 +
            df['soil_nitrogen'] / 20 * 0.3 +
            df['soil_phosphorus'] / 10 * 0.2 +
            df['soil_potassium'] / 50 * 0.2
        )

        # Optimal conditions indicators
        df['temp_optimal'] = ((df['temperature_avg'] >= 18) & (df['temperature_avg'] <= 30)).astype(int)
        df['ph_optimal'] = ((df['soil_ph'] >= 6.0) & (df['soil_ph'] <= 7.5)).astype(int)

        # Interaction terms
        df['temp_precip_interaction'] = df['temperature_avg'] * df['total_precipitation']
        df['gdd_precip_ratio'] = df['growing_degree_days'] / (df['total_precipitation'] + 1)

        return df


class YieldPredictionML:
    """Machine Learning models for crop yield prediction."""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_features(self, df, target_col='yield_tons_per_hectare'):
        """Prepare features for machine learning."""
        df_ml = df.copy()

        # One-hot encode categorical variables
        categorical_cols = ['crop', 'region']
        df_encoded = pd.get_dummies(df_ml, columns=categorical_cols, prefix=categorical_cols)

        # Separate features and target
        if target_col in df_encoded.columns:
            y = df_encoded[target_col]
            X = df_encoded.drop(columns=[target_col])
        else:
            X = df_encoded
            y = None

        self.feature_names = X.columns.tolist()
        return X, y

    def train_models(self, X, y, test_size=0.2):
        """Train multiple ML models for yield prediction."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models_config = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }

        # Train and evaluate models
        results = {}

        for name, model in models_config.items():
            logger.info(f"Training {name}...")

            # Use scaled features for linear models, original for tree-based
            if name in ['linear_regression', 'ridge']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test

            # Train model
            model.fit(X_train_model, y_train)

            # Make predictions
            y_pred = model.predict(X_test_model)

            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }

            logger.info(f"{name} - RMSE: {np.sqrt(mse):.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

        self.models = results
        return X_test, y_test, results

    def plot_model_comparison(self):
        """Plot comparison of different models."""
        if not self.models:
            logger.error("No models trained yet.")
            return

        model_names = list(self.models.keys())
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        r2_scores = [self.models[name]['r2'] for name in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # RMSE comparison
        ax1.bar(model_names, rmse_scores)
        ax1.set_title('Model RMSE Comparison')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)

        # R2 comparison
        ax2.bar(model_names, r2_scores)
        ax2.set_title('Model R² Comparison')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model_name='random_forest'):
        """Plot feature importance for tree-based models."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found.")
            return

        model = self.models[model_name]['model']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features

            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance ({model_name})')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            logger.error(f"Model {model_name} does not have feature_importances_ attribute.")

    def predict_yield(self, input_data, model_name='random_forest'):
        """Predict yield for new input data."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found.")
            return None

        model = self.models[model_name]['model']

        # Prepare input data
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Apply same preprocessing
        X, _ = self.prepare_features(input_df)

        # Use scaled features for linear models
        if model_name in ['linear_regression', 'ridge']:
            X = self.scaler.transform(X)

        prediction = model.predict(X)
        return prediction

    def save_models(self, model_dir):
        """Save trained models."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for name, model_info in self.models.items():
            model_path = model_dir / f'{name}_yield_model.pkl'
            joblib.dump(model_info['model'], model_path)

        # Save scaler and feature names
        joblib.dump(self.scaler, model_dir / 'yield_scaler.pkl')
        joblib.dump(self.feature_names, model_dir / 'yield_features.pkl')

        logger.info(f"Models saved to {model_dir}")


class YieldTimeSeriesModel:
    """Time series models for yield prediction."""

    def __init__(self):
        self.arima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()

    def prepare_time_series_data(self, df, crop_type, region):
        """Prepare time series data for a specific crop and region."""
        # Filter data
        ts_data = df[(df['crop'] == crop_type) & (df['region'] == region)].copy()
        ts_data = ts_data.sort_values('year')

        # Create time series
        ts = ts_data.set_index('year')['yield_tons_per_hectare']
        return ts

    def train_arima_model(self, ts_data, order=(1, 1, 1)):
        """Train ARIMA model for time series forecasting."""
        try:
            self.arima_model = ARIMA(ts_data, order=order)
            self.arima_fitted = self.arima_model.fit()

            logger.info(f"ARIMA model trained with order {order}")
            logger.info(f"AIC: {self.arima_fitted.aic:.2f}")

            return self.arima_fitted

        except Exception as e:
            logger.error(f"Failed to train ARIMA model: {e}")
            return None

    def create_lstm_model(self, input_shape):
        """Create LSTM model for yield prediction."""
        model = Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])

        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model

    def prepare_lstm_data(self, ts_data, lookback=5):
        """Prepare data for LSTM training."""
        # Normalize data
        scaled_data = self.scaler.fit_transform(ts_data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def train_lstm_model(self, ts_data, lookback=5, epochs=50):
        """Train LSTM model for time series forecasting."""
        X, y = self.prepare_lstm_data(ts_data, lookback)

        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create and train model
        self.lstm_model = self.create_lstm_model((lookback, 1))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )

        logger.info("LSTM model training completed")
        return history

    def forecast_arima(self, steps=5):
        """Forecast using ARIMA model."""
        if self.arima_fitted is None:
            logger.error("ARIMA model not trained.")
            return None

        forecast = self.arima_fitted.forecast(steps=steps)
        conf_int = self.arima_fitted.get_forecast(steps=steps).conf_int()

        return forecast, conf_int

    def forecast_lstm(self, ts_data, steps=5, lookback=5):
        """Forecast using LSTM model."""
        if self.lstm_model is None:
            logger.error("LSTM model not trained.")
            return None

        # Get last sequence
        scaled_data = self.scaler.fit_transform(ts_data.values.reshape(-1, 1))
        last_sequence = scaled_data[-lookback:]

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, lookback, 1)
            pred = self.lstm_model.predict(X_pred, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred[0, 0]

        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

    def plot_forecasts(self, ts_data, arima_forecast=None, lstm_forecast=None):
        """Plot historical data and forecasts."""
        plt.figure(figsize=(12, 8))

        # Plot historical data
        plt.plot(ts_data.index, ts_data.values, label='Historical Data', marker='o')

        # Plot ARIMA forecast
        if arima_forecast is not None:
            forecast_years = range(ts_data.index[-1] + 1, ts_data.index[-1] + 1 + len(arima_forecast[0]))
            plt.plot(forecast_years, arima_forecast[0], label='ARIMA Forecast', marker='s')

        # Plot LSTM forecast
        if lstm_forecast is not None:
            forecast_years = range(ts_data.index[-1] + 1, ts_data.index[-1] + 1 + len(lstm_forecast))
            plt.plot(forecast_years, lstm_forecast, label='LSTM Forecast', marker='^')

        plt.title('Yield Forecasting')
        plt.xlabel('Year')
        plt.ylabel('Yield (tons/hectare)')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    """Main function to demonstrate yield prediction pipeline."""
    logger.info("Starting Yield Prediction Pipeline")

    # Initialize components
    weather_collector = WeatherDataCollector()
    data_processor = YieldDataProcessor()
    ml_predictor = YieldPredictionML()
    ts_predictor = YieldTimeSeriesModel()

    # Create sample yield data
    logger.info("Creating sample yield dataset...")
    yield_df = data_processor.create_sample_yield_data(n_samples=2000)

    # Engineer features
    logger.info("Engineering features...")
    yield_df_processed = data_processor.engineer_features(yield_df)

    # Machine Learning approach
    logger.info("Training ML models...")
    X, y = ml_predictor.prepare_features(yield_df_processed)
    X_test, y_test, results = ml_predictor.train_models(X, y)

    # Plot model comparison
    ml_predictor.plot_model_comparison()
    ml_predictor.plot_feature_importance()

    # Time series approach
    logger.info("Training time series models...")
    wheat_ts = ts_predictor.prepare_time_series_data(yield_df, 'wheat', 'North')

    if len(wheat_ts) >= 10:  # Need sufficient data for time series
        ts_predictor.train_arima_model(wheat_ts)
        ts_predictor.train_lstm_model(wheat_ts)

        # Make forecasts
        arima_forecast = ts_predictor.forecast_arima(steps=5)
        lstm_forecast = ts_predictor.forecast_lstm(wheat_ts, steps=5)

        # Plot forecasts
        ts_predictor.plot_forecasts(wheat_ts, arima_forecast, lstm_forecast)

    # Save models
    ml_predictor.save_models("data/models")

    logger.info("Yield Prediction Pipeline completed!")


if __name__ == "__main__":
    main()

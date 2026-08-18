"""
Feature Engineering Module for Air Quality Forecasting.
Computes lag features, rolling statistics, calendar encodings, and weather interaction terms.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.config import LAG_HOURS, ROLLING_WINDOWS, TARGET_COL


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar and cyclical temporal features from timestamp.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def create_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL, lags: List[int] = LAG_HOURS) -> pd.DataFrame:
    """
    Create historical lag features for the target pollutant and weather parameters.
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)

    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    # Key weather lags
    for w_col in ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]:
        if w_col in df.columns:
            df[f"{w_col}_lag_24h"] = df[w_col].shift(24)

    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = TARGET_COL, windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Compute rolling statistics (mean, std, max, min) for the target variable.
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)

    for window in windows:
        roll = df[target_col].shift(1).rolling(window=window, min_periods=1)
        df[f"{target_col}_roll_mean_{window}h"] = roll.mean()
        df[f"{target_col}_roll_std_{window}h"] = roll.std().fillna(0)
        df[f"{target_col}_roll_max_{window}h"] = roll.max()
        df[f"{target_col}_roll_min_{window}h"] = roll.min()

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physical domain interaction features (e.g. wind dispersion, humidity-stagnation index).
    """
    df = df.copy()
    if "wind_speed_10m" in df.columns and "relative_humidity_2m" in df.columns:
        # High humidity + low wind speed = atmospheric inversion / stagnation hazard
        df["stagnation_index"] = df["relative_humidity_2m"] / (df["wind_speed_10m"] + 0.1)

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        # Thermal-moisture index
        df["temp_humidity_product"] = df["temperature_2m"] * df["relative_humidity_2m"]

    return df


def generate_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature pipeline application.
    """
    df = df.copy()
    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)

    return df


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """
    Extract all numeric feature column names, excluding non-predictive IDs and target.
    """
    exclude_cols = [
        "timestamp",
        "location",
        "date_time",
        "country",
        "city",
        "street",
        "url",
        TARGET_COL,
        "pm10",
        "european_aqi",
        "us_aqi",
        "us_aqi_calculated",
        "predicted_pm2_5",
        "predicted_us_aqi",
        "aqi_category",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
    ]
    features = [col for col in df.columns if col not in exclude_cols and not pd.api.types.is_string_dtype(df[col])]
    return features

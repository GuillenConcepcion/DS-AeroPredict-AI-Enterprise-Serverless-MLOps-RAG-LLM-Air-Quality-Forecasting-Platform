"""
Configuration module using Pydantic BaseSettings for Air Quality Forecasting MLOps System.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"
DOCS_DIR = BASE_DIR / "docs"
SECRETS_DIR = BASE_DIR / ".secrets"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SECRETS_DIR.mkdir(parents=True, exist_ok=True)

# Load secrets from hidden .secrets/secrets.env folder (excluded from Git/third parties)
if (SECRETS_DIR / "secrets.env").exists():
    load_dotenv(SECRETS_DIR / "secrets.env")
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".secrets/secrets.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Sensor Location Metadata
    country: str = Field(default="Sweden", alias="COUNTRY")
    city: str = Field(default="Stockholm", alias="CITY")
    street: str = Field(default="Central Station IoT", alias="STREET")
    url: str = Field(default="https://waqi.info/", alias="URL")
    latitude: float = Field(default=59.3293, alias="LATITUDE")
    longitude: float = Field(default=18.0686, alias="LONGITUDE")

    # Cloud Secrets
    hopsworks_project_name: str = Field(default="air_quality_prediction", alias="HOPSWORKS_PROJECT_NAME")
    hopsworks_api_key: str = Field(default="", alias="HOPSWORKS_API_KEY")
    aqicn_api_key: str = Field(default="", alias="AQICN_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")


# Global Settings Instance
settings = Settings()

# Hopsworks Feature Store Configuration
HOPSWORKS_PROJECT = settings.hopsworks_project_name
HOPSWORKS_API_KEY = settings.hopsworks_api_key

FEATURE_GROUP_NAME = "air_quality_hourly_fg"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "air_quality_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "air_quality_xgboost_model"

# Pre-defined Sensor Coordinates Lookup by City
CITY_COORDINATES: Dict[str, Dict[str, Any]] = {
    "Stockholm": {
        "city": "Stockholm",
        "country": "Sweden",
        "station_name": "Central Station IoT",
        "latitude": 59.3293,
        "longitude": 18.0686,
    },
    "Dublin": {
        "city": "Dublin",
        "country": "Ireland",
        "station_name": "Dublin City Center",
        "latitude": 53.3498,
        "longitude": -6.2603,
    },
    "Madrid": {
        "city": "Madrid",
        "country": "Spain",
        "station_name": "Madrid Centro",
        "latitude": 40.4168,
        "longitude": -3.7038,
    },
}

LOCATIONS = CITY_COORDINATES

DEFAULT_LOCATION = settings.city

# Data Fetching Endpoints
AQ_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_WEATHER_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

TARGET_COL = "pm2_5"

MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
}

AQ_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "european_aqi",
    "us_aqi",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

# Feature Engineering Settings
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [6, 24, 72, 168]
FORECAST_DAYS = 7
FORECAST_HORIZON_HOURS = 168

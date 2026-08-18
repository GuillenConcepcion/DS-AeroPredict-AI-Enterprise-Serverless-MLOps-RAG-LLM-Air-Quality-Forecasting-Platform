"""
Data Fetcher Module: Ingests historical and real-time air quality & weather measurements from Open-Meteo REST APIs,
enriches DataFrames with Pydantic sensor metadata (country, city, street, url), and performs dataset completeness audits.
"""
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from src.config import (
    settings,
    CITY_COORDINATES,
    AQ_ENDPOINT,
    WEATHER_ENDPOINT,
    ARCHIVE_WEATHER_ENDPOINT,
    AQ_VARIABLES,
    WEATHER_VARIABLES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dataset_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Evaluates dataset completeness by summarizing missing values (isna().sum()) per column.
    """
    missing_summary = df.isna().sum()
    total_rows = len(df)
    logger.info(f"--- Dataset Completeness Audit ({total_rows} total rows) ---")
    for col, null_cnt in missing_summary.items():
        if null_cnt > 0:
            logger.info(f"  • Column '{col}': {null_cnt} missing values ({(null_cnt/total_rows)*100:.2f}%)")
    return missing_summary


class AirQualityDataFetcher:
    def __init__(self, city_name: str = None, location_name: str = None):
        target_location = location_name or city_name or settings.city
        self.city = target_location
        self.country = settings.country
        self.street = settings.street
        self.url = settings.url

        # Retrieve lat/lon based on city/location
        if target_location in CITY_COORDINATES:
            coords = CITY_COORDINATES[target_location]
            self.lat = coords["latitude"]
            self.lon = coords["longitude"]
            self.country = coords.get("country", self.country)
            self.street = coords.get("station_name", self.street)
        else:
            self.lat = settings.latitude
            self.lon = settings.longitude

    def fetch_historical_air_quality(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical air quality measurements and add sensor helper columns (country, city, street, url).
        """
        logger.info(f"Fetching historical AQ data for city='{self.city}' ({start_date} to {end_date})...")
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ",".join(AQ_VARIABLES),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        response = requests.get(AQ_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        df_aq = pd.DataFrame(data["hourly"])
        df_aq["time"] = pd.to_datetime(df_aq["time"], utc=True)
        df_aq.rename(columns={"time": "timestamp"}, inplace=True)

        # Add Pydantic Metadata Helper Columns
        df_aq["country"] = self.country
        df_aq["city"] = self.city
        df_aq["street"] = self.street
        df_aq["url"] = self.url

        return df_aq

    def fetch_historical_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather measurements using city lat/lon.
        """
        logger.info(f"Fetching historical weather data for city='{self.city}' ({start_date} to {end_date})...")
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ",".join(WEATHER_VARIABLES),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        response = requests.get(ARCHIVE_WEATHER_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        df_w = pd.DataFrame(data["hourly"])
        df_w["time"] = pd.to_datetime(df_w["time"], utc=True)
        df_w.rename(columns={"time": "timestamp"}, inplace=True)
        df_w["city"] = self.city

        return df_w

    def fetch_recent_data(self, past_days: int = 90) -> pd.DataFrame:
        """
        Fetch and join Air Quality data with Weather features for the same date & city.
        """
        logger.info(f"Fetching recent data ({past_days} days) for city='{self.city}'...")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=past_days)

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        df_aq = self.fetch_historical_air_quality(start_str, end_str)
        df_weather = self.fetch_historical_weather(start_str, end_str)

        # Join Air Quality data with Weather features on timestamp and city
        df_merged = pd.merge(df_aq, df_weather, on=["timestamp", "city"], how="outer")
        df_merged.sort_values("timestamp", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        # Completeness Check
        check_dataset_completeness(df_merged)

        return df_merged

    def fetch_weather_forecast(self, forecast_days: int = 3) -> pd.DataFrame:
        """
        Fetch weather forecast for upcoming N days for batch inference.
        """
        logger.info(f"Fetching {forecast_days}-day weather forecast for city='{self.city}'...")
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ",".join(WEATHER_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        response = requests.get(WEATHER_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["country"] = self.country
        df["city"] = self.city
        df["street"] = self.street
        df.url = self.url
        return df


def calculate_us_aqi_pm25(pm25_val: float) -> int:
    if pd.isna(pm25_val) or pm25_val < 0:
        return 0

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25_val <= c_high:
            return int(round(((i_high - i_low) / (c_high - c_low)) * (pm25_val - c_low) + i_low))

    return 500

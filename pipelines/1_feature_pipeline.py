"""
Pipeline 1: Feature Pipeline
Ingests air quality and weather data, engineers lag & rolling features, and writes to Feature Store.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from src.config import DEFAULT_LOCATION, LOCATIONS
from src.data_fetcher import AirQualityDataFetcher, calculate_us_aqi_pm25
from src.features import generate_feature_pipeline
from src.hopsworks_utils import FeatureStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_feature_pipeline(location_name: str = DEFAULT_LOCATION, days_back: int = 180):
    logger.info(f"=== Running Feature Pipeline for location: {location_name} (Lookback: {days_back} days) ===")

    fetcher = AirQualityDataFetcher(location_name=location_name)
    raw_df = fetcher.fetch_recent_data(past_days=days_back)

    if raw_df.empty:
        logger.error("No data fetched. Exiting feature pipeline.")
        return

    logger.info(f"Fetched {len(raw_df)} raw records. Engineering features...")
    featured_df = generate_feature_pipeline(raw_df)

    # Compute US AQI from PM2.5
    if "pm2_5" in featured_df.columns:
        featured_df["us_aqi_calculated"] = featured_df["pm2_5"].apply(calculate_us_aqi_pm25)

    fs_manager = FeatureStoreManager()
    fs_manager.save_feature_group(featured_df)

    logger.info("=== Feature Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Location name")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data")
    args = parser.parse_args()

    run_feature_pipeline(location_name=args.location, days_back=args.days)

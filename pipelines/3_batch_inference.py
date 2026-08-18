"""
Pipeline 3: Batch Inference Pipeline
Generates future 72-hour AQI & PM2.5 forecasts and computes drift monitoring metrics on recent ground truth.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from src.config import DEFAULT_LOCATION, DATA_DIR, FORECAST_HORIZON_HOURS, MODEL_NAME, TARGET_COL
from src.data_fetcher import AirQualityDataFetcher, calculate_us_aqi_pm25
from src.features import generate_feature_pipeline, get_feature_names
from src.hopsworks_utils import FeatureStoreManager, ModelRegistryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_batch_inference(location_name: str = DEFAULT_LOCATION):
    logger.info(f"=== Running Batch Inference Pipeline for location: {location_name} ===")

    fs_manager = FeatureStoreManager()
    mr_manager = ModelRegistryManager(fs_manager)

    try:
        model = mr_manager.load_model(MODEL_NAME)
    except FileNotFoundError:
        logger.error("Model not found! Running training pipeline first...")
        import importlib
        training_mod = importlib.import_module("pipelines.2_training_pipeline")
        training_mod.run_training_pipeline()
        model = mr_manager.load_model(MODEL_NAME)

    fetcher = AirQualityDataFetcher(location_name=location_name)

    # 1. Fetch recent historical context (for lags) + upcoming 7-day weather forecast
    recent_df = fetcher.fetch_recent_data(past_days=14)
    weather_forecast_df = fetcher.fetch_weather_forecast(forecast_days=7)

    # Merge forecast timestamps onto combined structure
    combined_df = pd.concat([recent_df, weather_forecast_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    combined_df.sort_values("timestamp", inplace=True)

    # 2. Engineer features
    featured_df = generate_feature_pipeline(combined_df)
    feature_cols = get_feature_names(featured_df)

    # 3. Filter forecast horizon rows (future timestamps for 7 days)
    current_utc = datetime.now(timezone.utc)
    forecast_df = featured_df[featured_df["timestamp"] > current_utc].copy()

    if forecast_df.empty:
        # Fallback to last N rows if current_utc mismatch
        forecast_df = featured_df.tail(FORECAST_HORIZON_HOURS).copy()

    # Fill NaNs in features with forward/backward fill for forecast continuity
    X_forecast = forecast_df[feature_cols].ffill().bfill().fillna(0)

    # 4. Generate PM2.5 Predictions
    predicted_pm25 = model.predict(X_forecast)
    predicted_pm25 = np.clip(predicted_pm25, a_min=0, a_max=None)

    forecast_df["predicted_pm2_5"] = predicted_pm25
    forecast_df["predicted_us_aqi"] = forecast_df["predicted_pm2_5"].apply(calculate_us_aqi_pm25)

    # AQI Categorization
    def get_aqi_category(aqi_val: int) -> str:
        if aqi_val <= 50:
            return "Good"
        elif aqi_val <= 100:
            return "Moderate"
        elif aqi_val <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_val <= 200:
            return "Unhealthy"
        elif aqi_val <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    forecast_df["aqi_category"] = forecast_df["predicted_us_aqi"].apply(get_aqi_category)

    # 5. Model Monitoring & Drift Tracking on recent historical ground truth
    historical_eval = featured_df.dropna(subset=[TARGET_COL]).tail(48).copy()
    if len(historical_eval) > 0:
        X_eval = historical_eval[feature_cols].ffill().fillna(0)
        hist_preds = model.predict(X_eval)
        hist_preds = np.clip(hist_preds, a_min=0, a_max=None)
        recent_mae = float(round(np.mean(np.abs(historical_eval[TARGET_COL].values - hist_preds)), 3))
    else:
        recent_mae = 0.0

    # 6. Generate 7 Daily Aggregated Summary Predictions
    forecast_df["forecast_date"] = pd.to_datetime(forecast_df["timestamp"]).dt.date
    daily_summaries = []
    for f_date, group in forecast_df.groupby("forecast_date"):
        avg_pm25 = float(round(group["predicted_pm2_5"].mean(), 2))
        avg_aqi = int(round(group["predicted_us_aqi"].mean()))
        daily_summaries.append({
            "date": str(f_date),
            "predicted_pm2_5_mean": avg_pm25,
            "predicted_us_aqi_mean": avg_aqi,
            "aqi_category": get_aqi_category(avg_aqi),
            "max_pm2_5": float(round(group["predicted_pm2_5"].max(), 2)),
            "min_pm2_5": float(round(group["predicted_pm2_5"].min(), 2)),
        })

    # Output predictions JSON for UI serving
    results = {
        "location": location_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_days": 7,
        "recent_monitoring_mae": recent_mae,
        "current_pm2_5": float(round(recent_df[TARGET_COL].dropna().iloc[-1], 2)) if not recent_df.empty else 10.0,
        "current_us_aqi": calculate_us_aqi_pm25(recent_df[TARGET_COL].dropna().iloc[-1]) if not recent_df.empty else 40,
        "daily_forecast": daily_summaries,
        "forecast": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "predicted_pm2_5": float(round(row["predicted_pm2_5"], 2)),
                "predicted_us_aqi": int(row["predicted_us_aqi"]),
                "aqi_category": row["aqi_category"],
                "temperature": float(round(row.get("temperature_2m", 15.0), 1)),
                "humidity": float(round(row.get("relative_humidity_2m", 60.0), 1)),
                "wind_speed": float(round(row.get("wind_speed_10m", 5.0), 1)),
            }
            for _, row in forecast_df.iterrows()
        ],
    }

    # Save to data/ and docs/ (GitHub Pages root)
    out_json_path = DATA_DIR / "latest_predictions.json"
    docs_json_path = Path(__file__).resolve().parent.parent / "docs" / "latest_predictions.json"
    docs_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(docs_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 7. Generate Plotly Forecast Graph PNG for GitHub Pages
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.plot(forecast_df["timestamp"], forecast_df["predicted_pm2_5"], color="#2c3e50", linewidth=2, label="Predicted PM2.5 (7-Day Forecast)")
        plt.title(f"7-Day Air Quality PM2.5 Forecast ({location_name})", fontsize=14, fontweight="bold")
        plt.xlabel("Forecast Time")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        fig_png_path = docs_json_path.parent / "pm25_forecast.png"
        plt.savefig(fig_png_path, dpi=150)
        plt.close()
        logger.info(f"Generated 7-day forecast graph PNG at {fig_png_path}")
    except Exception as e:
        logger.warning(f"Could not render matplotlib graph PNG: {e}")

    logger.info(f"=== Batch Inference Completed! Generated {len(daily_summaries)} daily & {len(results['forecast'])} hourly predictions at {out_json_path} ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Location name")
    args = parser.parse_args()

    run_batch_inference(location_name=args.location)

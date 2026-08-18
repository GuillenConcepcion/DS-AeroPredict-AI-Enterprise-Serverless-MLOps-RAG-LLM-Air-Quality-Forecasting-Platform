"""
Hopsworks Feature Store, Model Registry Connector, and Secret Storage Manager.
"""
import os
import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

from src.config import (
    settings,
    HOPSWORKS_PROJECT,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_NAME,
    DATA_DIR,
    MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureStoreManager:
    def __init__(self, api_key: str = HOPSWORKS_API_KEY, project_name: str = HOPSWORKS_PROJECT):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None
        self._connected = False

        if self.api_key:
            try:
                import hopsworks
                logger.info(f"Connecting to Hopsworks Feature Store project '{self.project_name}'...")
                self.project = hopsworks.login(api_key_value=self.api_key, project=self.project_name)
                self.fs = self.project.get_feature_store()
                self._connected = True
                logger.info("Successfully connected to Hopsworks Feature Store!")
                self.store_pydantic_secrets()
            except Exception as e:
                logger.warning(f"Could not connect to Hopsworks ({e}). Operating in Local Parquet Fallback Mode.")
        else:
            logger.info("No HOPSWORKS_API_KEY found. Operating in Local Parquet Fallback Mode.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def store_pydantic_secrets(self) -> None:
        """
        Stores country, city, street, url, HOPSWORKS_API_KEY, and AQICN_API_KEY as secrets in Hopsworks.
        Subsequent pipelines can read these secrets directly without local .env dependency.
        """
        if not self.is_connected or self.project is None:
            return

        try:
            secret_api = self.project.get_secret_api()
            secrets_map = {
                "COUNTRY": settings.country,
                "CITY": settings.city,
                "STREET": settings.street,
                "URL": settings.url,
                "HOPSWORKS_API_KEY": settings.hopsworks_api_key,
                "AQICN_API_KEY": settings.aqicn_api_key,
            }

            for secret_name, secret_val in secrets_map.items():
                if secret_val:
                    try:
                        secret_api.create_secret(secret_name, secret_val)
                        logger.info(f"Registered secret '{secret_name}' in Hopsworks.")
                    except Exception:
                        pass  # Secret already exists
        except Exception as e:
            logger.warning(f"Could not sync secrets to Hopsworks: {e}")

    def get_secret(self, secret_name: str, fallback_default: str = "") -> str:
        """
        Retrieves a secret from Hopsworks Secrets Manager if connected, or falls back to local Settings.
        """
        if self.is_connected and self.project is not None:
            try:
                secret_api = self.project.get_secret_api()
                sec = secret_api.get_secret(secret_name)
                return sec.value
            except Exception:
                pass
        return getattr(settings, secret_name.lower(), fallback_default)

    def save_feature_group(self, df: pd.DataFrame, primary_key: list = ["timestamp", "city"]) -> None:
        """
        Write feature dataframe (including country, city, street, url metadata) to Hopsworks Feature Group or Local Parquet Storage.
        """
        df_clean = df.copy()
        if "timestamp" in df_clean.columns:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

        if self.is_connected and self.fs is not None:
            try:
                logger.info(f"Upserting data into Hopsworks Feature Group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
                fg = self.fs.get_or_create_feature_group(
                    name=FEATURE_GROUP_NAME,
                    version=FEATURE_GROUP_VERSION,
                    primary_key=primary_key,
                    event_time="timestamp",
                    description="Hourly IoT Air Quality, Weather features, and Pydantic Metadata",
                    online_enabled=True,
                )
                fg.insert(df_clean, write_options={"wait_for_job": False})
                logger.info("Feature group updated successfully in Hopsworks!")
                return
            except Exception as e:
                logger.error(f"Failed to write to Hopsworks Feature Group: {e}. Falling back to local storage.")

        # Local Parquet Fallback
        local_path = DATA_DIR / "air_quality_features.parquet"
        logger.info(f"Saving feature group locally to {local_path}...")
        if local_path.exists():
            existing_df = pd.read_parquet(local_path)
            combined_df = pd.concat([existing_df, df_clean]).drop_duplicates(subset=primary_key, keep="last")
            combined_df.sort_values("timestamp", inplace=True)
            combined_df.to_parquet(local_path, index=False)
        else:
            df_clean.to_parquet(local_path, index=False)
        logger.info(f"Feature group saved locally ({len(df_clean)} rows).")

    def read_feature_group(self) -> pd.DataFrame:
        """
        Read historical feature dataset from Hopsworks or Local Parquet Storage.
        """
        if self.is_connected and self.fs is not None:
            try:
                logger.info(f"Reading Hopsworks Feature Group '{FEATURE_GROUP_NAME}'...")
                fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
                df = fg.read()
                logger.info(f"Read {len(df)} records from Hopsworks Feature Group.")
                return df
            except Exception as e:
                logger.warning(f"Error reading from Hopsworks Feature Group: {e}. Falling back to local storage.")

        # Local Parquet Fallback
        local_path = DATA_DIR / "air_quality_features.parquet"
        if local_path.exists():
            logger.info(f"Reading features from local file {local_path}...")
            return pd.read_parquet(local_path)
        else:
            raise FileNotFoundError("No feature data found in Hopsworks or Local Storage. Run 1_feature_pipeline.py first.")


class ModelRegistryManager:
    def __init__(self, feature_manager: FeatureStoreManager):
        self.fm = feature_manager

    def save_model(
        self, model: Any, metrics: dict, feature_view: Any = None, model_name: str = MODEL_NAME
    ) -> str:
        import shutil

        model_dir = MODELS_DIR / "air_quality_model"
        images_dir = model_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        local_model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, local_model_path)

        # Save at MODELS_DIR root for backward compatibility
        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")
        logger.info(f"Model saved locally at {local_model_path}")

        # Copy evaluation images (feature_importance.png, pm25_hindcast.png) to images/ subfolder
        for img_file in MODELS_DIR.glob("*.png"):
            shutil.copy(img_file, images_dir / img_file.name)
            logger.info(f"Attached evaluation image {img_file.name} to {images_dir}")

        if self.fm.is_connected and self.fm.project is not None:
            try:
                mr = self.fm.project.get_model_registry()
                logger.info(f"Registering model '{model_name}' to Hopsworks Model Registry...")

                # Standardize metric keys to match Hopsworks UI schema (MSE, r2, MAE)
                formatted_metrics = {
                    "MSE": float(metrics.get("rmse", 0.0) ** 2) if "rmse" in metrics else float(metrics.get("mse", 0.0)),
                    "r2": float(metrics.get("r2", 0.0)),
                    "MAE": float(metrics.get("mae", 0.0)),
                }

                create_kwargs = {
                    "name": model_name,
                    "description": "Air Quality (PM2.5) predictor.",
                    "metrics": formatted_metrics,
                }
                if feature_view is not None:
                    create_kwargs["feature_view"] = feature_view

                hw_model = mr.python.create_model(**create_kwargs)
                hw_model.save(str(model_dir))
                logger.info("Model, schema lineage, and evaluation images registered successfully in Hopsworks!")
            except Exception as e:
                logger.warning(f"Failed to register model in Hopsworks: {e}")

        return str(local_model_path)

    def load_model(self, model_name: str = MODEL_NAME) -> Any:
        paths = [
            MODELS_DIR / "air_quality_model" / f"{model_name}.joblib",
            MODELS_DIR / f"{model_name}.joblib",
            MODELS_DIR / "stockholm_air_quality_xgboost.joblib",
            MODELS_DIR / "stockholm_air_quality_xgboost.pkl",
        ]
        for path in paths:
            if path.exists():
                logger.info(f"Loading model from local artifact {path}...")
                return joblib.load(path)

        raise FileNotFoundError(f"Model file for {model_name} not found. Please run 2_training_pipeline.py first.")

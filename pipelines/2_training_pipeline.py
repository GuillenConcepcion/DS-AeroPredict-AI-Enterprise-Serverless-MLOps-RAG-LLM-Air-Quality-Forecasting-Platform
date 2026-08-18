"""
Pipeline 2: Training Pipeline
Fetches feature view, splits time-series data, trains model, evaluates metrics, and registers model artifact.
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from src.config import DATA_DIR, MODELS_DIR, MODEL_NAME
from src.hopsworks_utils import FeatureStoreManager, ModelRegistryManager
from src.models import (
    prepare_train_test_split,
    train_air_quality_model,
    compare_and_select_best_model,
    get_feature_importances,
    generate_evaluation_plots,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline():
    logger.info("=== Running Model Training Pipeline (Multi-Algorithm Benchmark) ===")

    fs_manager = FeatureStoreManager()
    df = fs_manager.read_feature_group()

    if df is None or df.empty:
        logger.error("No feature data found. Aborting training.")
        return

    logger.info(f"Loaded dataset with {len(df)} rows. Performing time-series split...")
    X_train, y_train, X_test, y_test, feature_names = prepare_train_test_split(df)

    # Benchmark LightGBM, XGBoost, RandomForest & Select Best Model
    model, metrics, df_benchmark = compare_and_select_best_model(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)

    # Generate Hopsworks Model Registry Evaluation Images
    generate_evaluation_plots(model, feature_names, y_test, y_pred, MODELS_DIR)

    # Save Feature Importances
    df_imp = get_feature_importances(model, feature_names)
    imp_path = MODELS_DIR / "feature_importances.json"
    df_imp.head(15).to_json(imp_path, orient="records", indent=2)
    logger.info(f"Top 5 predictive features:\n{df_imp.head(5)}")

    # Save metrics JSON
    metrics_path = MODELS_DIR / "latest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Register Model Artifact
    mr_manager = ModelRegistryManager(fs_manager)
    model_path = mr_manager.save_model(model, metrics, MODEL_NAME)

    logger.info(f"=== Model Training Pipeline Completed! Registered model at {model_path} ===")


if __name__ == "__main__":
    run_training_pipeline()

"""
Model Training, Evaluation, and Inference Module for Air Quality Prediction.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from typing import Tuple, Dict, Any, List
import logging

from src.config import MODEL_PARAMS, TARGET_COL
from src.features import get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_train_test_split(
    df: pd.DataFrame, target_col: str = TARGET_COL, test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Split time-series data chronologically into training and validation sets.
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.dropna(subset=[target_col], inplace=True)

    feature_cols = get_feature_names(df)

    # Drop rows with NaNs in feature columns caused by lag computation
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    split_idx = int(len(df_clean) * (1 - test_ratio))
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Dataset split: Train shape={X_train.shape}, Test shape={X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


def train_air_quality_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, algorithm: str = "lightgbm"
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a specified machine learning algorithm (lightgbm, xgboost, catboost, random_forest).
    """
    logger.info(f"Training {algorithm.upper()} model for PM2.5 prediction...")

    if algorithm.lower() == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif algorithm.lower() == "catboost":
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        except ImportError:
            logger.warning("CatBoost not installed. Falling back to LightGBM.")
            model = lgb.LGBMRegressor(**MODEL_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    elif algorithm.lower() == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    else:
        # Default: LightGBM
        model = lgb.LGBMRegressor(**MODEL_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

    preds = model.predict(X_val)
    preds = np.clip(preds, a_min=0, a_max=None)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    metrics = {
        "algorithm": algorithm,
        "mae": float(round(mae, 4)),
        "rmse": float(round(rmse, 4)),
        "r2": float(round(r2, 4)),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_val),
    }

    logger.info(f"Model ({algorithm}) Complete | MAE: {mae:.3f} ug/m3 | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return model, metrics


def compare_and_select_best_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    """
    Train and evaluate multiple algorithms (LightGBM, XGBoost, CatBoost, RandomForest)
    and select the best performing model based on MAE/RMSE.
    """
    algorithms = ["lightgbm", "xgboost", "random_forest"]

    results = []
    trained_models = {}

    for algo in algorithms:
        try:
            m, met = train_air_quality_model(X_train, y_train, X_val, y_val, algorithm=algo)
            trained_models[algo] = (m, met)
            results.append(met)
        except Exception as e:
            logger.warning(f"Could not benchmark algorithm '{algo}': {e}")

    df_benchmark = pd.DataFrame(results).sort_values("mae", ascending=True)
    logger.info(f"\n=== MULTI-ALGORITHM BENCHMARK RESULTS ===\n{df_benchmark[['algorithm', 'mae', 'rmse', 'r2']]}\n")

    best_algo = df_benchmark.iloc[0]["algorithm"]
    best_model, best_metrics = trained_models[best_algo]
    logger.info(f"🏆 Selected Best Model: '{best_algo.upper()}' (MAE: {best_metrics['mae']:.3f})")

    return best_model, best_metrics, df_benchmark


def get_feature_importances(model: lgb.LGBMRegressor, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate and return feature importances sorted descending.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp.sort_values("importance", ascending=False, inplace=True)
    return df_imp


def generate_evaluation_plots(
    model: lgb.LGBMRegressor,
    feature_names: List[str],
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Generate feature_importance.png and pm25_hindcast.png evaluation images for Hopsworks Model Registry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fi_path = output_dir / "feature_importance.png"
    hindcast_path = output_dir / "pm25_hindcast.png"

    # 1. Feature Importance Plot
    df_imp = get_feature_importances(model, feature_names).head(12)
    plt.figure(figsize=(9, 5))
    plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="#1f77b4")
    plt.title("Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()

    # 2. PM2.5 Hindcast Evaluation Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(y_test), label="Actual PM2.5", color="#111111", linewidth=1.5)
    plt.plot(y_pred, label="Predicted PM2.5", color="#e74c3c", linestyle="--", linewidth=1.5)
    plt.title("PM2.5 Predicted vs Actual (Hindcast Validation)", fontsize=14, fontweight="bold")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.xlabel("Validation Timesteps (Hours)")
    plt.axhspan(0, 50, color="green", alpha=0.1, label="Good (0-50)")
    plt.axhspan(50, 100, color="yellow", alpha=0.1, label="Moderate (50-100)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(hindcast_path, dpi=150)
    plt.close()

    logger.info(f"Saved evaluation plots: {fi_path.name}, {hindcast_path.name}")
    return fi_path, hindcast_path

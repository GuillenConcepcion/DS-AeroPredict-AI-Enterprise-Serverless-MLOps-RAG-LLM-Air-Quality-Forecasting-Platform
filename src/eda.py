"""
Exploratory Data Analysis (EDA) Module for Air Quality & Meteorological Time Series Data.
Includes statistical audits, stationarity testing (ADF), seasonal decomposition, and correlation analysis.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_DIR, DOCS_DIR, TARGET_COL, AQ_VARIABLES, WEATHER_VARIABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AirQualityEDA:
    """
    Programmatic Exploratory Data Analysis suite for Time-Series Air Quality Data.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COL):
        self.df = df.copy()
        self.target_col = target_col
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df.sort_values("timestamp", inplace=True)

    def run_summary_statistics(self) -> pd.DataFrame:
        """
        Computes summary statistics (mean, std, min, 25%, median, 75%, max, skewness, kurtosis).
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        stats = numeric_df.describe().T
        stats["skewness"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurt()
        stats["missing_count"] = numeric_df.isna().sum()
        stats["missing_percentage"] = (numeric_df.isna().sum() / len(numeric_df)) * 100.0
        return stats

    def run_stationarity_adf_test(self, col: str = TARGET_COL) -> Dict[str, Any]:
        """
        Performs Augmented Dickey-Fuller (ADF) test for time-series stationarity.
        """
        from statsmodels.tsa.stattools import adfuller

        series = self.df[col].dropna()
        if len(series) < 20:
            return {"status": "error", "message": "Insufficient data points for ADF test."}

        result = adfuller(series, autolag="AIC")
        is_stationary = bool(result[1] < 0.05)

        res_dict = {
            "variable": col,
            "adf_statistic": float(round(result[0], 4)),
            "p_value": float(round(result[1], 4)),
            "used_lags": int(result[2]),
            "n_observations": int(result[3]),
            "critical_values_1pct": float(round(result[4]["1%"], 4)),
            "critical_values_5pct": float(round(result[4]["5%"], 4)),
            "is_stationary": is_stationary,
            "interpretation": "Series is Stationary (p < 0.05)" if is_stationary else "Series is Non-Stationary (p >= 0.05)"
        }
        logger.info(f"ADF Test [{col}]: p-value={res_dict['p_value']} -> {res_dict['interpretation']}")
        return res_dict

    def generate_correlation_matrix_plot(self, save_path: Path) -> Path:
        """
        Generates and saves a correlation heatmap for air quality and weather variables.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cols_to_corr = [c for c in AQ_VARIABLES + WEATHER_VARIABLES if c in self.df.columns]
        corr = self.df[cols_to_corr].corr()

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Pearson Correlation")
        plt.xticks(range(len(cols_to_corr)), cols_to_corr, rotation=45, ha="right", fontsize=9)
        plt.yticks(range(len(cols_to_corr)), cols_to_corr, fontsize=9)
        plt.title("Air Quality & Meteorology Feature Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f"Saved correlation plot at {save_path}")
        return save_path

    def generate_seasonal_decomposition_plot(self, col: str = TARGET_COL, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Performs classical seasonal decomposition (Trend, Seasonality, Residuals).
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        if "timestamp" in self.df.columns:
            ts_df = self.df.set_index("timestamp")[col].dropna()
        else:
            ts_df = self.df[col].dropna()

        if len(ts_df) < 48:
            logger.warning("Insufficient observations for seasonal decomposition.")
            return None

        # Resample to hourly if needed
        ts_df = ts_df.resample("1h").mean().ffill().bfill()
        decomp = seasonal_decompose(ts_df, model="additive", period=24)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        ax1.plot(decomp.observed, color="#2c3e50")
        ax1.set_ylabel("Observed")
        ax1.set_title(f"Seasonal Decomposition for {col}", fontsize=13, fontweight="bold")

        ax2.plot(decomp.trend, color="#e74c3c")
        ax2.set_ylabel("Trend")

        ax3.plot(decomp.seasonal, color="#27ae60")
        ax3.set_ylabel("Seasonal (24h)")

        ax4.scatter(decomp.resid.index, decomp.resid, color="#7f8c8d", s=5)
        ax4.set_ylabel("Residuals")

        plt.tight_layout()

        out_path = save_path or (DATA_DIR / f"{col}_seasonal_decomp.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()

        logger.info(f"Saved seasonal decomposition plot at {out_path}")
        return out_path

    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Runs complete EDA pipeline and returns structured report dictionary.
        """
        stats_df = self.run_summary_statistics()
        adf_result = self.run_stationarity_adf_test(self.target_col)
        corr_plot_path = self.generate_correlation_matrix_plot(DATA_DIR / "eda_correlation_matrix.png")
        decomp_plot_path = self.generate_seasonal_decomposition_plot(self.target_col, DATA_DIR / "eda_seasonal_decomposition.png")

        # Copy plots to docs directory for web serving
        import shutil
        docs_corr = DOCS_DIR / "eda_correlation_matrix.png"
        docs_decomp = DOCS_DIR / "eda_seasonal_decomposition.png"
        if corr_plot_path and corr_plot_path.exists():
            shutil.copy(corr_plot_path, docs_corr)
        if decomp_plot_path and decomp_plot_path.exists():
            shutil.copy(decomp_plot_path, docs_decomp)

        report = {
            "n_records": len(self.df),
            "n_features": len(self.df.columns),
            "target_variable": self.target_col,
            "stationarity_adf": adf_result,
            "summary_stats": stats_df.to_dict(orient="index"),
            "correlation_plot_path": str(corr_plot_path),
            "decomposition_plot_path": str(decomp_plot_path) if decomp_plot_path else None
        }

        report_json_path = DATA_DIR / "eda_report.json"
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"=== EDA Report generated successfully at {report_json_path} ===")
        return report


if __name__ == "__main__":
    from src.hopsworks_utils import FeatureStoreManager
    fs = FeatureStoreManager()
    df_features = fs.read_feature_group()
    eda = AirQualityEDA(df_features)
    eda.generate_eda_report()

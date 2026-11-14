"""
src/analysis/residuals.py
==========================
Residual / error analysis for trained models.

Produces four diagnostic artefacts per model:

1. Error distribution histogram  — raw prediction error (y - ŷ)
2. Predicted vs Actual scatter   — with identity line
3. QQ plot                       — standardised residuals vs Normal quantiles
4. Hurdle calibration curve      — binned P(delay) vs actual delay rate
   (only for the hurdle model outputs)

All plots are saved as PNGs under ``results/``.

Public API
----------
  load_test_predictions(spark, paths, task, condition) -> pd.DataFrame
  plot_residuals(pdf, task, condition, algo, paths) -> None
  plot_hurdle_calibration(hurdle_csv_path, paths) -> None
  run_residual_analysis(spark, paths) -> None
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import (
    BTS_FEATURE_COLS,
    PATHS,
    WEATHER_FEATURE_COLS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONDITION_COLS = {
    "bts_only":    BTS_FEATURE_COLS,
    "bts_weather": WEATHER_FEATURE_COLS,
}

_TARGET_COL = {
    "departure": "DepDelay",
    "arrival":   "ArrDelay",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_predictions(
    spark: SparkSession,
    model_dir: Path,
    feature_set_dir: Path,
    target_col: str,
    sample_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Load a saved PipelineModel, run it on the test partition, return a small
    pandas DataFrame with columns [actual, predicted, error].
    """
    pipeline_model = PipelineModel.load(str(model_dir))
    test_df = (
        spark.read.parquet(str(feature_set_dir))
        .filter(F.col("data_split") == "test")
        .sample(fraction=sample_frac, seed=42)
    )
    preds = pipeline_model.transform(test_df)
    pdf = (
        preds
        .select(
            F.col(target_col).alias("actual"),
            F.col("prediction").alias("predicted"),
        )
        .toPandas()
        .dropna()
    )
    pdf["error"] = pdf["actual"] - pdf["predicted"]
    return pdf


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def _plot_error_distribution(errors: pd.Series, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors.clip(-120, 120), bins=80, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="--", linewidth=1.2,
               label=f"Mean error = {errors.mean():.1f} min")
    ax.set_xlabel("Prediction Error (actual − predicted, minutes)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pred_vs_actual(pdf: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pdf["actual"], pdf["predicted"], alpha=0.15, s=4, color="#4c72b0")
    lim = max(abs(pdf["actual"].quantile(0.99)), abs(pdf["predicted"].quantile(0.99)))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Delay (min)")
    ax.set_ylabel("Predicted Delay (min)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_qq(errors: pd.Series, title: str, out_path: Path) -> None:
    std_errors = (errors - errors.mean()) / (errors.std() + 1e-9)
    sample = std_errors.sample(min(5000, len(std_errors)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    (osm, osr), (slope, intercept, _) = stats.probplot(sample, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="#4c72b0", label="Residuals")
    x_line = np.array([min(osm), max(osm)])
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.2, label="Normal reference")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hurdle calibration curve
# ---------------------------------------------------------------------------

def plot_hurdle_calibration(
    spark: SparkSession,
    paths: dict,
    task: str = "departure",
    n_bins: int = 10,
) -> None:
    """
    Calibration curve for the hurdle classifier stage.

    Loads the saved hurdle classifier, scores the test partition, bins
    P(delay > 0) into *n_bins* buckets, and plots the mean predicted
    probability vs the actual fraction of delayed flights in each bucket.
    """
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.sql import types as T

    clf_dir = paths["models"] / f"{task}_hurdle_classifier"
    if not clf_dir.exists():
        logger.warning("Hurdle classifier not found at %s — skipping calibration.", clf_dir)
        return

    get_prob1 = F.udf(lambda v: float(v[1]), T.DoubleType())

    target_col = _TARGET_COL[task]
    feature_dir = paths["features"] / f"{task}_bts_weather"

    clf_model = PipelineModel.load(str(clf_dir))
    test_df = (
        spark.read.parquet(str(feature_dir))
        .filter(F.col("data_split") == "test")
    )
    preds = clf_model.transform(test_df)
    pdf = (
        preds
        .select(
            get_prob1("probability").alias("p_delayed"),
            F.when(F.col(target_col) > 0, 1.0).otherwise(0.0).alias("actually_delayed"),
        )
        .toPandas()
        .dropna()
    )

    pdf["bin"] = pd.cut(pdf["p_delayed"], bins=n_bins, labels=False)
    cal = pdf.groupby("bin").agg(
        mean_pred=("p_delayed", "mean"),
        actual_rate=("actually_delayed", "mean"),
        count=("p_delayed", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cal["mean_pred"], cal["actual_rate"], s=cal["count"] / cal["count"].max() * 200,
               color="#4c72b0", zorder=3, label="Calibration bins")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted P(delay > 0)")
    ax.set_ylabel("Actual Fraction Delayed")
    ax.set_title(f"Hurdle Classifier Calibration — {task}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = paths["results"] / f"hurdle_calibration_{task}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Hurdle calibration plot saved → %s", out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_residual_analysis(
    spark: SparkSession,
    paths: dict = PATHS,
    tasks: list = None,
    conditions: list = None,
    algorithms: list = None,
    sample_frac: float = 0.1,
) -> None:
    """
    Generate all residual diagnostic plots for every available saved model.

    Uses a 10 % sample of the test partition to keep it fast.
    """
    tasks      = tasks      or ["departure", "arrival"]
    conditions = conditions or ["bts_only", "bts_weather"]
    algorithms = algorithms or ["RandomForest", "GBTRegressor"]

    for task in tasks:
        target_col  = _TARGET_COL[task]
        feature_dir_map = {
            c: paths["features"] / f"{task}_{c}" for c in conditions
        }

        for condition in conditions:
            feature_dir = feature_dir_map[condition]
            if not feature_dir.exists():
                logger.warning("Feature set %s not found — skipping.", feature_dir)
                continue

            for algo in algorithms:
                model_dir = paths["models"] / f"{task}_{condition}_{algo}"
                if not model_dir.exists():
                    logger.warning("Model %s not found — skipping.", model_dir)
                    continue

                logger.info("Residual analysis: %s | %s | %s", task, condition, algo)
                try:
                    pdf = _get_predictions(
                        spark, model_dir, feature_dir, target_col,
                        sample_frac=sample_frac,
                    )
                except Exception as exc:
                    logger.error("Failed to get predictions: %s", exc)
                    continue

                stem = f"{task}_{condition}_{algo}"
                title_base = f"{task.title()} | {condition} | {algo}"

                _plot_error_distribution(
                    pdf["error"],
                    f"Error Distribution — {title_base}",
                    paths["results"] / f"residuals_dist_{stem}.png",
                )
                _plot_pred_vs_actual(
                    pdf,
                    f"Predicted vs Actual — {title_base}",
                    paths["results"] / f"pred_vs_actual_{stem}.png",
                )
                _plot_qq(
                    pdf["error"],
                    f"QQ Plot — {title_base}",
                    paths["results"] / f"qq_{stem}.png",
                )
                logger.info("  Residual plots saved for %s", stem)

        # Hurdle calibration curve
        plot_hurdle_calibration(spark, paths, task=task)

    logger.info("Residual analysis complete.")

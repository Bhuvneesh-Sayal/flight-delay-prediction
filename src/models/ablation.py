"""
src/models/ablation.py
=======================
Controlled ablation experiment runner.

Trains three regression algorithms (LinearRegression, RandomForest, GBT)
across two feature conditions (bts_only, bts_weather) and two prediction
targets (departure delay, arrival delay), plus a train-mean baseline for
each target.

Each algorithm is tuned with k-fold CrossValidator over a parameter grid
(see LR_PARAM_GRID / RF_PARAM_GRID / GBT_PARAM_GRID in config.settings) and
the best model is evaluated on both the test and validation partitions and
saved to disk.

Metrics reported: MAE, RMSE, R²  (test and validation partitions).

Public API
----------
  run_ablation(spark, paths, bts_feature_cols, weather_feature_cols) -> pd.DataFrame
"""

import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config.settings import (
    BTS_FEATURE_COLS,
    CV_FOLDS,
    GBT_PARAM_GRID,
    GBT_PARAMS,
    LR_PARAM_GRID,
    LR_PARAMS,
    PATHS,
    RF_PARAM_GRID,
    RF_PARAMS,
    WEATHER_FEATURE_COLS,
)
from src.models.pipeline import build_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Evaluators (re-used across all runs)
# ---------------------------------------------------------------------------
_eval_mae  = RegressionEvaluator(metricName="mae")
_eval_rmse = RegressionEvaluator(metricName="rmse")
_eval_r2   = RegressionEvaluator(metricName="r2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(preds: DataFrame, target_col: str) -> tuple:
    """Return (MAE, RMSE, R²) of the 'prediction' column against *target_col*."""
    mae  = _eval_mae.setLabelCol(target_col).evaluate(preds)
    rmse = _eval_rmse.setLabelCol(target_col).evaluate(preds)
    r2   = _eval_r2.setLabelCol(target_col).evaluate(preds)
    return mae, rmse, r2


def _build_param_grid(estimator, grid: dict) -> list:
    """
    Turn a {param_name: [values…]} dict into a ParamGridBuilder grid bound to
    *estimator*.
    """
    builder = ParamGridBuilder()
    for name, values in grid.items():
        builder = builder.addGrid(getattr(estimator, name), list(values))
    return builder.build()


def _result_row(task, condition, algo, test_metrics, val_metrics) -> dict:
    return {
        "Task":          task,
        "Condition":     condition,
        "Algorithm":     algo,
        "Test_MAE_min":  round(test_metrics[0], 2),
        "Test_RMSE_min": round(test_metrics[1], 2),
        "Test_R2":       round(test_metrics[2], 4),
        "Val_MAE_min":   round(val_metrics[0], 2),
        "Val_RMSE_min":  round(val_metrics[1], 2),
        "Val_R2":        round(val_metrics[2], 4),
    }


def _mean_baseline(
    train_df: DataFrame,
    test_df: DataFrame,
    val_df: DataFrame,
    target_col: str,
) -> tuple:
    """Predict the train-set mean of *target_col* for every row."""
    mean_val = train_df.agg(F.avg(target_col)).first()[0]
    logger.info("    MeanBaseline: train mean of %s = %.4f", target_col, mean_val)

    test_metrics = _score(
        test_df.withColumn("prediction", F.lit(float(mean_val))), target_col
    )
    val_metrics = _score(
        val_df.withColumn("prediction", F.lit(float(mean_val))), target_col
    )
    return test_metrics, val_metrics


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_ablation(
    spark: SparkSession,
    paths: dict = PATHS,
    bts_feature_cols: list = BTS_FEATURE_COLS,
    weather_feature_cols: list = WEATHER_FEATURE_COLS,
) -> pd.DataFrame:
    """
    Run the full ablation matrix and return a summary DataFrame.

    The summary is also saved to results/controlled_ablation_summary.csv.
    """
    # Factories so every (task, condition) gets a fresh, unshared estimator.
    ablation_models = {
        "LinearRegression": (lambda: LinearRegression(**LR_PARAMS),        LR_PARAM_GRID),
        "RandomForest":     (lambda: RandomForestRegressor(**RF_PARAMS),   RF_PARAM_GRID),
        "GBTRegressor":     (lambda: GBTRegressor(**GBT_PARAMS),           GBT_PARAM_GRID),
    }

    results: list = []

    for task, target_col in [
        ("departure", "DepDelay"),
        ("arrival",   "ArrDelay"),
    ]:
        for condition, p_cols in [
            ("bts_only",    bts_feature_cols),
            ("bts_weather", weather_feature_cols),
        ]:
            dataset_name = f"{task}_{condition}"
            ds = spark.read.parquet(str(paths["features"] / dataset_name))

            train_df = ds.filter(F.col("data_split") == "train").cache()
            val_df   = ds.filter(F.col("data_split") == "validation").cache()
            test_df  = ds.filter(F.col("data_split") == "test").cache()

            # ----- Mean baseline (before any ML model) -----
            logger.info("  %s | %s | MeanBaseline…", task.upper(), condition)
            base_test, base_val = _mean_baseline(train_df, test_df, val_df, target_col)
            results.append(
                _result_row(task, condition, "MeanBaseline", base_test, base_val)
            )

            # ----- ML models with CV + hyper-parameter tuning -----
            for algo_name, (make_estimator, param_grid) in ablation_models.items():
                logger.info("  %s | %s | %s…", task.upper(), condition, algo_name)

                estimator = make_estimator()
                estimator.setLabelCol(target_col).setFeaturesCol("features")

                pipe = build_pipeline(p_cols, estimator, df=train_df)
                grid = _build_param_grid(estimator, param_grid)

                cv = CrossValidator(
                    estimator=pipe,
                    estimatorParamMaps=grid,
                    evaluator=RegressionEvaluator(
                        labelCol=target_col, metricName="mae"
                    ),
                    numFolds=CV_FOLDS,
                    seed=42,
                )
                cv_model = cv.fit(train_df)
                model = cv_model.bestModel

                best_idx = min(
                    range(len(cv_model.avgMetrics)),
                    key=lambda i: cv_model.avgMetrics[i],
                )
                logger.info(
                    "    best CV MAE=%.3f params=%s",
                    cv_model.avgMetrics[best_idx],
                    {p.name: v for p, v in grid[best_idx].items()},
                )

                test_metrics = _score(model.transform(test_df), target_col)
                val_metrics  = _score(model.transform(val_df),  target_col)
                results.append(
                    _result_row(task, condition, algo_name, test_metrics, val_metrics)
                )

                # Persist the best model (task is included so departure and
                # arrival models do not overwrite each other).
                model_path = str(
                    paths["models"] / f"{task}_{condition}_{algo_name}"
                )
                model.write().overwrite().save(model_path)
                logger.info("    model saved → %s", model_path)

            train_df.unpersist()
            val_df.unpersist()
            test_df.unpersist()

    summary = pd.DataFrame(results)
    out = paths["results"] / "controlled_ablation_summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Ablation summary saved → %s", out)
    return summary

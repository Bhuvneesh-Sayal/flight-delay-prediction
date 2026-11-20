"""
src/models/hurdle.py
=====================
Two-stage hurdle model for skew-aware delay prediction.

Stage 1 (classifier)
    Logistic Regression: P(delay > DELAY_BINARY_THRESHOLD)

Stage 2 (regressor, trained on delayed flights only)
    GBT Regressor on log1p(delay), trained on rows where
    delay > DELAY_POSITIVE_THRESHOLD

Combined prediction
    E[delay] = P(delayed) × (exp(stage2_prediction) − 1)

The binary label is built on the fly from the raw delay column so both
stages share a consistent boundary definition (see config.settings).

Metrics: Stage-1 ROC-AUC, combined MAE, RMSE, R²  (test partition).
Both fitted stage models are saved under paths["models"].

Public API
----------
  run_hurdle(spark, paths, weather_feature_cols) -> pd.DataFrame
"""

import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config.settings import (
    DELAY_BINARY_THRESHOLD,
    DELAY_POSITIVE_THRESHOLD,
    FULL_FEATURE_COLS,
    GBT_PARAMS,
    PATHS,
    WEATHER_FEATURE_COLS,
)
from src.models.pipeline import build_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# UDF: extract P(class=1) from a probability vector
# ---------------------------------------------------------------------------
_get_prob1 = F.udf(lambda v: float(v[1]), T.DoubleType())

# ---------------------------------------------------------------------------
# Shared evaluators — NOTE: these are created fresh per call in
# _run_single_task to avoid mutating shared state across conditions.
# ---------------------------------------------------------------------------

# Name of the on-the-fly binary label column
_CLS_LABEL = "delay_positive"


# ---------------------------------------------------------------------------
# Single-task hurdle
# ---------------------------------------------------------------------------

def _run_single_task(
    spark: SparkSession,
    task: str,
    raw_target: str,
    log_target: str,
    paths: dict,
    weather_feature_cols: list,
    condition: str = "bts_weather",
) -> dict:
    """Train and evaluate the two-stage hurdle model for one task."""
    logger.info("Training hurdle model: %s (%s)…", task.upper(), condition)

    # Local evaluators — avoid mutating shared module-level state
    eval_mae  = RegressionEvaluator(metricName="mae")
    eval_rmse = RegressionEvaluator(metricName="rmse")
    eval_r2   = RegressionEvaluator(metricName="r2")

    ds = spark.read.parquet(str(paths["features"] / f"{task}_{condition}"))
    ds = ds.withColumn(
        _CLS_LABEL,
        F.when(F.col(raw_target) > DELAY_BINARY_THRESHOLD, 1.0).otherwise(0.0),
    )
    train_data = ds.filter(F.col("data_split") == "train").cache()
    test_data  = ds.filter(F.col("data_split") == "test").cache()

    # ----- Stage 1: binary classifier -----
    clf = LogisticRegression(
        labelCol=_CLS_LABEL, featuresCol="features", maxIter=25
    )
    clf_pipe  = build_pipeline(weather_feature_cols, clf, df=train_data)
    clf_model = clf_pipe.fit(train_data)
    clf_preds = clf_model.transform(test_data)

    auc_eval   = BinaryClassificationEvaluator(
        labelCol=_CLS_LABEL, metricName="areaUnderROC"
    )
    stage1_auc = auc_eval.evaluate(clf_preds)

    clf_path = str(paths["models"] / f"hurdle_{task}_{condition}_classifier")
    clf_model.write().overwrite().save(clf_path)
    logger.info("  classifier saved → %s", clf_path)

    # ----- Stage 2: regressor on delayed subset -----
    train_delayed = train_data.filter(F.col(raw_target) > DELAY_POSITIVE_THRESHOLD)
    reg = GBTRegressor(labelCol=log_target, featuresCol="features", **GBT_PARAMS)
    reg_pipe  = build_pipeline(weather_feature_cols, reg, df=train_delayed)
    reg_model = reg_pipe.fit(train_delayed)

    reg_path = str(paths["models"] / f"hurdle_{task}_{condition}_regressor")
    reg_model.write().overwrite().save(reg_path)
    logger.info("  regressor saved → %s", reg_path)

    test_scored = reg_model.transform(test_data)
    test_scored = test_scored.withColumn(
        "raw_pred_delay", F.expm1(F.col("prediction"))
    )

    # ----- Combine: E[delay] = P(delayed) × E[delay | delayed] -----
    clf_prob_df = clf_preds.select(
        "flight_key",
        _get_prob1("probability").alias("p_delayed"),
    )
    final = (
        test_scored
        .join(clf_prob_df, on="flight_key")
        .withColumn("final_prediction", F.col("p_delayed") * F.col("raw_pred_delay"))
    )

    two_stage_mae  = eval_mae.setLabelCol(raw_target).setPredictionCol("final_prediction").evaluate(final)
    two_stage_rmse = eval_rmse.setLabelCol(raw_target).setPredictionCol("final_prediction").evaluate(final)
    two_stage_r2   = eval_r2.setLabelCol(raw_target).setPredictionCol("final_prediction").evaluate(final)

    train_data.unpersist()
    test_data.unpersist()

    return {
        "Task":               task,
        "Condition":          condition,
        "Stage1_ROC_AUC":     round(stage1_auc,     4),
        "TwoStage_MAE_min":   round(two_stage_mae,  2),
        "TwoStage_RMSE_min":  round(two_stage_rmse, 2),
        "TwoStage_R2":        round(two_stage_r2,   4),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_hurdle(
    spark: SparkSession,
    paths: dict = PATHS,
    weather_feature_cols: list = WEATHER_FEATURE_COLS,
    full_feature_cols: list = FULL_FEATURE_COLS,
) -> pd.DataFrame:
    """
    Train and evaluate two-stage hurdle models for departure and arrival,
    across both the bts_weather and full feature conditions.
    Returns a summary DataFrame; also saves to results/two_stage_hurdle_results.csv.
    """
    summary_rows: list = []

    conditions = [
        ("bts_weather", weather_feature_cols),
        ("full",        full_feature_cols),
    ]

    for task, raw_tgt, log_tgt in [
        ("departure", "DepDelay", "log_DepDelay"),
        ("arrival",   "ArrDelay", "log_ArrDelay"),
    ]:
        for condition, feat_cols in conditions:
            ds_path = paths["features"] / f"{task}_{condition}"
            if not ds_path.exists():
                logger.info("Feature set %s not found — skipping.", ds_path)
                continue
            row = _run_single_task(
                spark,
                task=task,
                raw_target=raw_tgt,
                log_target=log_tgt,
                paths=paths,
                weather_feature_cols=feat_cols,
                condition=condition,
            )
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    out = paths["results"] / "two_stage_hurdle_results.csv"
    summary.to_csv(out, index=False)
    logger.info("Hurdle results saved → %s", out)
    return summary

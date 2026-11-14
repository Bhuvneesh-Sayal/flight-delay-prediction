"""
main.py
=======
Entry point for the Flight Delay Prediction pipeline.

Usage
-----
    python main.py                      # run all steps
    python main.py --steps ingest prep  # run specific steps only

Steps (in order)
----------------
    ingest        Download BTS + NOAA raw data
    prep          Clean flights and standardise weather
    features      Feature engineering (UTC, congestion, weather join, labels)
    ablation      Controlled ablation experiment (12 models, includes baseline)
    hurdle        Two-stage hurdle model
    analysis      Feature importances + residual diagnostic plots
    benchmarks    Strong + weak scaling benchmarks (local, simulated cores)
    cluster_bench Inter-node network shuffle benchmarks on the Docker cluster
    stream        Spark Structured Streaming pipeline (scores live flight events)
"""

import argparse
import sys

# Only stdlib + logger imported at module level — everything else is lazy.
# This means `python main.py --steps ingest` does NOT import PySpark at all,
# avoiding the distutils / Python-version issue for steps that don't need Spark.
from config.settings import CLUSTER_MASTER_URL, PATHS, SPARK_CONFIG
from src.utils.logger import get_logger

ALL_STEPS = [
    "ingest", "prep", "features",
    "ablation", "hurdle", "analysis",
    "benchmarks", "cluster_bench", "stream",
]

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Spark factory  (imported lazily so non-Spark steps skip PySpark entirely)
# ---------------------------------------------------------------------------

def _create_spark():
    import pyspark
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("FlightDelayPrediction").master("local[*]")
    for k, v in SPARK_CONFIG.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark %s | PySpark %s", spark.version, pyspark.__version__)
    return spark


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flight Delay Prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps: {', '.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        metavar="STEP",
        help="Steps to run (default: all). Choices: %(choices)s",
    )
    return parser.parse_args()


def _banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run(steps: list) -> None:
    bts_csv_files = None
    noaa_paths = None
    flights_df = None
    weather_parquet_path = None

    needs_spark = any(s in steps for s in ["prep", "features", "ablation", "hurdle", "analysis", "stream"])
    spark = _create_spark() if needs_spark else None

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    if "ingest" in steps:
        _banner("STEP 1 — Data Ingestion")
        from src.data.ingestion import run_ingestion
        bts_csv_files, noaa_paths = run_ingestion()

    # If ingest was skipped but prep is requested, discover files already on disk
    if "prep" in steps and bts_csv_files is None:
        from src.data.ingestion import _discover_cached
        bts_csv_files, noaa_paths = _discover_cached()

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    if "prep" in steps:
        _banner("STEP 2 — Preprocessing")
        from src.data.preprocessing import run_preprocessing
        flights_df, weather_parquet_path = run_preprocessing(
            spark, bts_csv_files, noaa_paths
        )

    # ── Step 3: Feature engineering ────────────────────────────────────────
    if "features" in steps:
        if flights_df is None or weather_parquet_path is None:
            logger.error("'features' requires 'prep' to have run first.")
            sys.exit(1)
        _banner("STEP 3 — Feature Engineering")
        from src.features.engineering import run_feature_engineering
        run_feature_engineering(spark, flights_df, weather_parquet_path)

    # ── Step 4: Ablation ───────────────────────────────────────────────────
    if "ablation" in steps:
        _banner("STEP 4 — Ablation Experiments")
        from src.models.ablation import run_ablation
        summary = run_ablation(spark)
        logger.info("Ablation summary:\n%s", summary.to_string(index=False))

    # ── Step 5: Hurdle model ───────────────────────────────────────────────
    if "hurdle" in steps:
        _banner("STEP 5 — Two-Stage Hurdle Model")
        from src.models.hurdle import run_hurdle
        summary = run_hurdle(spark)
        logger.info("Hurdle summary:\n%s", summary.to_string(index=False))

    # ── Step 6: Analysis ───────────────────────────────────────────────────
    if "analysis" in steps:
        _banner("STEP 6 — Feature Importances & Residual Analysis")
        from src.analysis.feature_importance import run_feature_importance_analysis
        from src.analysis.residuals import run_residual_analysis
        run_feature_importance_analysis(paths=PATHS)
        run_residual_analysis(spark, paths=PATHS)

    # ── Step 7: Local scaling benchmarks ───────────────────────────────────
    if "benchmarks" in steps:
        _banner("STEP 7 — Scaling Benchmarks (local)")
        from src.benchmarks.scaling import plot_scaling_curves, run_scaling_benchmarks
        if spark is not None:
            spark.stop()
            spark = None
        bench_df = run_scaling_benchmarks()
        plot_scaling_curves(bench_df)
        logger.info("Benchmark results:\n%s", bench_df.to_string(index=False))

    # ── Step 8: Cluster benchmarks ─────────────────────────────────────────
    if "cluster_bench" in steps:
        _banner("STEP 8 — Cluster Scaling Benchmarks (Docker cluster)")
        from src.benchmarks.cluster_scaling import run_cluster_benchmarks
        if spark is not None:
            spark.stop()
            spark = None
        cluster_df = run_cluster_benchmarks(master_url=CLUSTER_MASTER_URL, paths=PATHS)
        logger.info("Cluster benchmark results:\n%s", cluster_df.to_string(index=False))

    # ── Step 9: Structured Streaming ───────────────────────────────────────
    if "stream" in steps:
        _banner("STEP 9 — Structured Streaming Pipeline")
        from src.streaming.pipeline import run_streaming_pipeline
        if spark is None:
            spark = _create_spark()
        run_streaming_pipeline(spark, paths=PATHS, timeout_secs=60)

    logger.info("Pipeline complete.")


def main() -> None:
    try:
        args = _parse_args()
        _run(args.steps)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()

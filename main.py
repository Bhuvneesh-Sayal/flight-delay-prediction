"""
main.py
=======
Entry point for the Flight Delay Prediction pipeline.

Usage
-----
    python main.py                           # run all steps
    python main.py --steps ingest prep       # specific steps
    python main.py --steps stream            # streaming (file source)
    python main.py --steps stream --kafka    # streaming (Kafka source)
    python main.py --steps produce           # replay BTS data into Kafka
    python main.py --steps produce --produce-rate 100   # 100 events/sec
    python main.py --steps produce --produce-limit 1000 # first 1000 rows

Steps (in order)
----------------
    ingest        Download BTS + NOAA raw data
    prep          Clean flights and standardise weather
    features      Feature engineering (UTC, congestion, weather join, labels)
    ablation      Controlled ablation experiment (18 models, includes baseline)
    hurdle        Two-stage hurdle model
    analysis      Feature importances + residual diagnostic plots
    benchmarks    Strong + weak scaling benchmarks (local, simulated cores)
    cluster_bench Inter-node network shuffle benchmarks on the Docker cluster
    produce       Replay historical BTS data into Kafka (simulate live feed)
    stream        Spark Structured Streaming pipeline
                  (use --kafka to consume from Kafka instead of files)
"""

import argparse
import os
import sys

from config.settings import CLUSTER_MASTER_URL, PATHS, SPARK_CONFIG
from src.utils.logger import get_logger

ALL_STEPS = [
    "ingest", "prep", "features",
    "ablation", "hurdle", "analysis",
    "benchmarks", "cluster_bench",
    "produce", "stream",
]

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Spark factory
# ---------------------------------------------------------------------------

def _create_spark(with_kafka: bool = False):
    import pyspark
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("FlightDelayPrediction")
    # Under spark-submit, the JVM is already running and spark-submit sets
    # PYSPARK_GATEWAY_PORT. Leave the master unset there so --master
    # (e.g. spark://spark-master:7077) takes effect; a master set in code
    # would override it. Plain `python main.py` still runs in local mode.
    if "PYSPARK_GATEWAY_PORT" not in os.environ:
        builder = builder.master("local[*]")
    for k, v in SPARK_CONFIG.items():
        builder = builder.config(k, v)

    if with_kafka:
        from config.settings import KAFKA_SPARK_PACKAGE
        # The Kafka SQL connector is loaded as a Spark package (downloaded on
        # first run, cached in ~/.ivy2 / ~/.m2 thereafter).
        builder = builder.config("spark.jars.packages", KAFKA_SPARK_PACKAGE)
        logger.info("Kafka SQL connector package: %s", KAFKA_SPARK_PACKAGE)

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
    parser.add_argument(
        "--kafka",
        action="store_true",
        default=False,
        help="For --steps stream: read from Kafka instead of local files.",
    )
    parser.add_argument(
        "--produce-rate",
        type=int,
        default=50,
        metavar="N",
        help="For --steps produce: max events published per second (default 50, 0=unlimited).",
    )
    parser.add_argument(
        "--produce-limit",
        type=int,
        default=0,
        metavar="N",
        help="For --steps produce: stop after N events (default 0 = all).",
    )
    return parser.parse_args()


def _banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run(steps: list, kafka: bool, produce_rate: int, produce_limit: int) -> None:
    bts_csv_files = None
    noaa_paths    = None
    storm_csv     = None
    flights_df    = None
    weather_parquet_path = None

    needs_spark = any(s in steps for s in
                      ["prep", "features", "ablation", "hurdle", "analysis", "stream"])
    spark = _create_spark(with_kafka=(kafka and "stream" in steps)) if needs_spark else None

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    if "ingest" in steps:
        _banner("STEP 1 — Data Ingestion")
        from src.data.ingestion import run_ingestion
        bts_csv_files, noaa_paths, storm_csv, _holidays = run_ingestion()

    # Discover cached files when ingest was skipped
    if "prep" in steps and bts_csv_files is None:
        from src.data.ingestion import _discover_cached
        bts_csv_files, noaa_paths, storm_csv, _holidays = _discover_cached()

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    if "prep" in steps:
        _banner("STEP 2 — Preprocessing")
        from src.data.preprocessing import run_preprocessing
        flights_df, weather_parquet_path = run_preprocessing(
            spark, bts_csv_files, noaa_paths, storm_csv_path=storm_csv,
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

    # ── Step 9: Kafka producer (replay BTS → Kafka) ─────────────────────────
    if "produce" in steps:
        _banner("STEP 9 — Kafka Producer (BTS replay)")
        from src.streaming.producer import run_producer
        from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_FLIGHT_TOPIC
        run_producer(
            paths=PATHS,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            topic=KAFKA_FLIGHT_TOPIC,
            rate_per_sec=produce_rate,
            limit=produce_limit,
        )

    # ── Step 10: Structured Streaming ──────────────────────────────────────
    if "stream" in steps:
        source = "kafka" if kafka else "file"
        _banner(f"STEP 10 — Structured Streaming Pipeline (source: {source})")
        from src.streaming.pipeline import run_streaming_pipeline
        if spark is None:
            spark = _create_spark(with_kafka=kafka)
        run_streaming_pipeline(
            spark,
            paths=PATHS,
            source=source,
            timeout_secs=-1,
        )

    logger.info("Pipeline complete.")


def main() -> None:
    try:
        args = _parse_args()
        _run(
            steps=args.steps,
            kafka=args.kafka,
            produce_rate=args.produce_rate,
            produce_limit=args.produce_limit,
        )
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

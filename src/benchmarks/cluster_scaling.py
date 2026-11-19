"""
src/benchmarks/cluster_scaling.py
===================================
Cluster scaling benchmark — submits workloads to a real Spark standalone
cluster (spark://spark-master:7077) to measure inter-node network shuffle
performance.

Unlike src/benchmarks/scaling.py (which uses local[P] to simulate cores),
this module connects to the Docker cluster, distributes data across all three
workers, and measures actual network shuffle overhead.

Benchmark workloads
-------------------
1. Shuffle-heavy aggregation (groupBy + multi-agg)
   — forces a full data redistribute across all workers
2. Sort-based benchmark (orderBy on a large key space)
   — stresses the network shuffle sort
3. Join benchmark (two-sided hash join of flight data)
   — measures inter-node join performance

Outputs
-------
  benchmarks_output/cluster_scaling_results.csv
  benchmarks_output/cluster_scaling_chart.png

Public API
----------
  run_cluster_benchmarks(master_url, paths, rows) -> pd.DataFrame
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import PATHS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default cluster URL — can be overridden via env var SPARK_MASTER_URL
import os
DEFAULT_MASTER_URL = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")


# ---------------------------------------------------------------------------
# Spark factory for cluster mode
# ---------------------------------------------------------------------------

def _create_cluster_spark(master_url: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName("FlightDelay_ClusterBenchmark")
        .master(master_url)
        # Give each executor 3g; with 3 workers × 2 cores = 6 total executor slots
        .config("spark.executor.memory",           "3g")
        .config("spark.executor.cores",            "2")
        .config("spark.driver.memory",             "2g")
        .config("spark.sql.shuffle.partitions",    "24")   # 4 × num executor cores
        .config("spark.sql.adaptive.enabled",      "true")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Individual benchmark workloads
# ---------------------------------------------------------------------------

def _bench_shuffle_agg(spark: SparkSession, parquet_path: str, rows: int) -> float:
    """GroupBy + multi-aggregation — forces full shuffle."""
    df = spark.read.parquet(parquet_path).limit(rows)
    start = time.perf_counter()
    _ = (
        df.groupBy("Origin", "Dest", "Reporting_Airline")
        .agg(
            F.count("flight_key").alias("n"),
            F.avg("DepDelay").alias("avg_dep_delay"),
            F.stddev("DepDelay").alias("std_dep_delay"),
            F.percentile_approx("DepDelay", 0.95).alias("p95_dep_delay"),
            F.avg("origin_congestion_2h").alias("avg_congestion"),
        )
        .collect()
    )
    return time.perf_counter() - start


def _bench_sort(spark: SparkSession, parquet_path: str, rows: int) -> float:
    """Global sort — stresses network shuffle sort stage."""
    df = spark.read.parquet(parquet_path).limit(rows)
    start = time.perf_counter()
    _ = (
        df.orderBy("FlightDate", "Origin", "flight_key")
        .select("flight_key", "Origin", "FlightDate", "DepDelay")
        .collect()
    )
    return time.perf_counter() - start


def _bench_join(spark: SparkSession, parquet_path: str, rows: int) -> float:
    """
    Self-join on (Origin, scheduled_month) to simulate a typical
    broadcast-less join between two large datasets.
    """
    df = spark.read.parquet(parquet_path).limit(rows)
    df_a = df.select(
        F.col("Origin").alias("orig_a"),
        F.col("scheduled_month").alias("month_a"),
        F.col("DepDelay").alias("dep_a"),
        F.col("flight_key").alias("key_a"),
    )
    df_b = df.select(
        F.col("Origin").alias("orig_b"),
        F.col("scheduled_month").alias("month_b"),
        F.col("DepDelay").alias("dep_b"),
        F.col("flight_key").alias("key_b"),
    )
    start = time.perf_counter()
    _ = (
        df_a.join(df_b,
                  (df_a.orig_a == df_b.orig_b) & (df_a.month_a == df_b.month_b),
                  "inner")
        .agg(F.count("*"), F.avg("dep_a"), F.avg("dep_b"))
        .collect()
    )
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_cluster_results(bench_df: pd.DataFrame, paths: dict) -> None:
    workloads = bench_df["Workload"].unique()
    rows_list = bench_df["Rows"].unique()

    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 5), sharey=False)
    if len(workloads) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, wl in zip(axes, workloads):
        sub = bench_df[bench_df["Workload"] == wl].sort_values("Rows")
        ax.bar(
            [str(r) for r in sub["Rows"]],
            sub["Time_Sec"],
            color=colors[: len(sub)],
            edgecolor="white",
        )
        ax.set_title(f"{wl}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Rows processed")
        ax.set_ylabel("Time (seconds)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Cluster Benchmark — 3-node Spark (6 executor cores total)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = paths["benchmarks"] / "cluster_scaling_chart.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Cluster benchmark chart saved → %s", out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_cluster_benchmarks(
    master_url: str = DEFAULT_MASTER_URL,
    paths: dict = PATHS,
    row_sizes: list = None,
) -> pd.DataFrame:
    """
    Run all three benchmark workloads at multiple data sizes against the
    cluster at *master_url*.

    Parameters
    ----------
    master_url : Spark standalone master URL, e.g. "spark://spark-master:7077"
    paths      : project paths dict
    row_sizes  : list of row counts to benchmark, default [100_000, 300_000, 465_000]

    Returns
    -------
    pd.DataFrame with columns: Workload, Rows, Time_Sec
    """
    row_sizes = row_sizes or [100_000, 300_000, 465_000]
    parquet_path = str(paths["features"] / "departure_bts_weather")

    logger.info("Connecting to cluster at %s…", master_url)
    spark = _create_cluster_spark(master_url)
    logger.info(
        "Connected. Executors: %d",
        spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size(),
    )

    workloads = {
        "ShuffleAgg": _bench_shuffle_agg,
        "GlobalSort": _bench_sort,
        "HashJoin":   _bench_join,
    }

    results: list = []

    for wl_name, wl_fn in workloads.items():
        for rows in row_sizes:
            logger.info("Running %s | rows=%d…", wl_name, rows)
            try:
                elapsed = wl_fn(spark, parquet_path, rows)
            except Exception as exc:
                import traceback
                logger.error("Workload %s failed for rows=%d: %s\n%s",
                             wl_name, rows, exc, traceback.format_exc())
                elapsed = float("nan")

            results.append({"Workload": wl_name, "Rows": rows, "Time_Sec": round(elapsed, 3)})
            logger.info("  → %.3f s", elapsed)

    spark.stop()

    bench_df = pd.DataFrame(results)
    out_csv = paths["benchmarks"] / "cluster_scaling_results.csv"
    bench_df.to_csv(out_csv, index=False)
    logger.info("Cluster benchmark results saved → %s", out_csv)

    _plot_cluster_results(bench_df, paths)
    return bench_df

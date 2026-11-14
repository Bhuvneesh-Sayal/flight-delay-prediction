"""
src/benchmarks/scaling.py
==========================
Strong and weak scaling benchmarks for the DA231o rubric.

Strong scaling (Amdahl's Law)
    Fixed dataset size (STRONG_SCALE_ROWS), vary executor cores P ∈ {1, 2, 4}.
    Measures speedup S(P) = T(1) / T(P).

Weak scaling (Gustafson's Law)
    Dataset grows linearly with cores (WEAK_SCALE_ROWS_PER_CORE × P),
    cores vary P ∈ {1, 2, 4}.
    Ideal: execution time stays constant.

The benchmark workload is a distributed groupBy + aggregation over the
departure_bts_weather feature set — representative of the shuffle-heavy
operations in the real pipeline.

Public API
----------
  run_scaling_benchmarks(paths, cores_list, strong_rows, weak_rows_per_core)
      -> pd.DataFrame
  plot_scaling_curves(bench_df, paths) -> None
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import (
    BENCHMARK_CORES,
    PATHS,
    STRONG_SCALE_ROWS,
    WEAK_SCALE_ROWS_PER_CORE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spark(app_name: str, cores: int) -> SparkSession:
    """Create a fresh local SparkSession with *cores* executor threads."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.sql.shuffle.partitions", str(cores * 2))
        .getOrCreate()
    )


def _benchmark_aggregation(spark: SparkSession, parquet_path: str, row_limit: int) -> float:
    """
    Run a groupBy aggregation on *row_limit* rows from *parquet_path*.
    Returns wall-clock elapsed seconds.
    """
    bench_df = spark.read.parquet(parquet_path).limit(row_limit)
    start = time.perf_counter()
    _ = (
        bench_df
        .groupBy("Origin", "Reporting_Airline")
        .agg(
            F.avg("DepDelay"),
            F.stddev("DepDelay"),
            F.count("flight_key"),
        )
        .collect()
    )
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Main benchmark runners
# ---------------------------------------------------------------------------

def _run_strong_scaling(
    parquet_path: str,
    cores_list: list,
    strong_rows: int,
) -> list:
    rows = []
    print("--- Strong Scaling (Amdahl's Law) ---")
    for p in cores_list:
        spark = _make_spark(f"StrongScaling_{p}", p)
        elapsed = _benchmark_aggregation(spark, parquet_path, strong_rows)
        rows.append({"Type": "Strong", "Cores": p, "Workload": strong_rows, "Time_Sec": elapsed})
        print(f"  Cores={p} | Rows={strong_rows:,} | Time={elapsed:.3f}s")
        spark.stop()
    return rows


def _run_weak_scaling(
    parquet_path: str,
    cores_list: list,
    rows_per_core: int,
) -> list:
    rows = []
    print("\n--- Weak Scaling (Gustafson's Law) ---")
    for p in cores_list:
        workload = p * rows_per_core
        spark = _make_spark(f"WeakScaling_{p}", p)
        elapsed = _benchmark_aggregation(spark, parquet_path, workload)
        rows.append({"Type": "Weak", "Cores": p, "Workload": workload, "Time_Sec": elapsed})
        print(f"  Cores={p} | Rows={workload:,} | Time={elapsed:.3f}s")
        spark.stop()
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scaling_benchmarks(
    paths: dict = PATHS,
    cores_list: list = BENCHMARK_CORES,
    strong_rows: int  = STRONG_SCALE_ROWS,
    weak_rows_per_core: int = WEAK_SCALE_ROWS_PER_CORE,
) -> pd.DataFrame:
    """
    Run strong and weak scaling benchmarks.
    Returns a DataFrame; also saves to benchmarks/scaling_benchmarks.csv.
    """
    parquet_path = str(paths["features"] / "departure_bts_weather")

    all_rows  = _run_strong_scaling(parquet_path, cores_list, strong_rows)
    all_rows += _run_weak_scaling(parquet_path, cores_list, weak_rows_per_core)

    bench_df = pd.DataFrame(all_rows)

    # Compute speedup for strong scaling only
    t1 = bench_df.loc[
        (bench_df["Type"] == "Strong") & (bench_df["Cores"] == 1), "Time_Sec"
    ].values[0]

    bench_df["Speedup"] = bench_df.apply(
        lambda r: round(t1 / r["Time_Sec"], 2) if r["Type"] == "Strong" else np.nan,
        axis=1,
    )

    out = paths["benchmarks"] / "scaling_benchmarks.csv"
    bench_df.to_csv(out, index=False)
    print(f"\nScaling benchmarks saved → {out}")
    return bench_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling_curves(bench_df: pd.DataFrame, paths: dict = PATHS) -> None:
    """
    Render Amdahl (strong) and Gustafson (weak) scaling charts and save
    to benchmarks/scaling_curves.png.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    strong_df = bench_df[bench_df["Type"] == "Strong"]
    ax1.plot(strong_df["Cores"], strong_df["Speedup"],
             marker="o", linewidth=2, color="#1f77b4", label="Observed Speedup")
    ax1.plot(strong_df["Cores"], strong_df["Cores"],
             "--", color="gray", label="Ideal Linear Speedup")
    ax1.set_title("Amdahl's Law: Strong Scaling Speedup", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Executor Cores (P)")
    ax1.set_ylabel("Speedup S(P) = T(1) / T(P)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    weak_df = bench_df[bench_df["Type"] == "Weak"]
    ax2.plot(weak_df["Cores"], weak_df["Time_Sec"],
             marker="s", linewidth=2, color="#2ca02c", label="Execution Time")
    ax2.set_title("Gustafson's Law: Weak Scaling Execution Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Executor Cores (P)")
    ax2.set_ylabel("Elapsed Time (seconds)")
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out = paths["benchmarks"] / "scaling_curves.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"Scaling curves saved → {out}")

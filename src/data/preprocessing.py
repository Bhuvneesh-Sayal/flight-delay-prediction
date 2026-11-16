"""
src/data/preprocessing.py
==========================
Cleans and standardises raw BTS flight data and raw NOAA weather data,
producing canonical Parquet files ready for feature engineering.

Public API
----------
  clean_flights(spark, bts_csv_files, paths, top_10_airports) -> DataFrame
  standardise_weather(noaa_paths, year, paths) -> DataFrame
  run_quality_checks(flights_df) -> None
  run_preprocessing(spark, bts_csv_files, noaa_paths, paths) -> tuple
"""

import shutil
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import (
    BTS_KEEP_COLS,
    NOAA_KEEP_COLS,
    PATHS,
    TOP_10_AIRPORTS,
    YEAR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BTS flight cleaning
# ---------------------------------------------------------------------------

def clean_flights(
    spark: SparkSession,
    bts_csv_files: list,
    paths: dict = PATHS,
    top_10_airports: list = TOP_10_AIRPORTS,
) -> object:
    """
    Read all BTS CSV files, apply the closed-cohort filter and type casts,
    persist the result as Snappy Parquet, and return the Spark DataFrame.

    Closed-cohort rule: both Origin *and* Dest must be in *top_10_airports*.
    Cancelled and diverted flights are excluded.
    Rows with null DepDelay or ArrDelay are excluded.
    """
    # --- detect tail-number column name (varies by BTS download vintage) ---
    sample_header = pd.read_csv(bts_csv_files[0], nrows=0).columns.tolist()
    tail_col = "Tail_Number" if "Tail_Number" in sample_header else "TailNum"

    keep_cols = [c if c != "Tail_Number" else tail_col for c in BTS_KEEP_COLS]

    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(bts_csv_files)
        .select(*keep_cols)
    )

    # Normalise tail-number column name
    if tail_col != "Tail_Number":
        raw = raw.withColumnRenamed(tail_col, "Tail_Number")

    flights = (
        raw
        .withColumn("FlightDate", F.to_date("FlightDate"))
        .withColumn("Origin",     F.upper(F.trim("Origin")))
        .withColumn("Dest",       F.upper(F.trim("Dest")))
        # Closed-cohort filter
        .filter(
            F.col("Origin").isin(top_10_airports) &
            F.col("Dest").isin(top_10_airports)
        )
        # Drop cancelled / diverted
        .filter(
            (F.coalesce(F.col("Cancelled"), F.lit(0)) == 0) &
            (F.coalesce(F.col("Diverted"),  F.lit(0)) == 0)
        )
        # Drop rows with missing delay values
        .filter(F.col("DepDelay").isNotNull() & F.col("ArrDelay").isNotNull())
        # Type casts
        .withColumn("DepDelay",       F.col("DepDelay").cast("double"))
        .withColumn("ArrDelay",       F.col("ArrDelay").cast("double"))
        .withColumn("CRSElapsedTime", F.col("CRSElapsedTime").cast("double"))
        .withColumn("CRSDepTime",     F.col("CRSDepTime").cast("int"))
        .withColumn("CRSArrTime",     F.col("CRSArrTime").cast("int"))
        # Delay-cause columns (null for on-time flights → fill with 0)
        .withColumn("CarrierDelay",      F.coalesce(F.col("CarrierDelay").cast("double"),      F.lit(0.0)))
        .withColumn("WeatherDelay",      F.coalesce(F.col("WeatherDelay").cast("double"),      F.lit(0.0)))
        .withColumn("NASDelay",          F.coalesce(F.col("NASDelay").cast("double"),          F.lit(0.0)))
        .withColumn("SecurityDelay",     F.coalesce(F.col("SecurityDelay").cast("double"),     F.lit(0.0)))
        .withColumn("LateAircraftDelay", F.coalesce(F.col("LateAircraftDelay").cast("double"), F.lit(0.0)))
        # Calendar features
        .withColumn("scheduled_month",       F.month("FlightDate"))
        .withColumn("scheduled_day_of_week", F.dayofweek("FlightDate"))
    )

    # Persist to Parquet for downstream reuse
    out = paths["processed"] / f"bts_cohort_{YEAR}.parquet"
    flights.write.mode("overwrite").option("compression", "snappy").parquet(str(out))
    flights = spark.read.parquet(str(out))

    logger.info("Cleaned cohort: %s rows → %s", f"{flights.count():,}", out)
    return flights


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

_NULL_RATE_COLS = ["DepDelay", "ArrDelay", "CRSElapsedTime", "Tail_Number"]
_DELAY_COLS = ["DepDelay", "ArrDelay"]


def run_quality_checks(flights_df) -> None:
    """
    Log data-quality metrics for the cleaned flights DataFrame:
      - total row count
      - null rate for DepDelay, ArrDelay, CRSElapsedTime, Tail_Number
      - delay distribution (mean, stddev, min, max, p50, p95) for DepDelay / ArrDelay
      - row counts per data_split (only if the column exists)
    """
    total = flights_df.count()
    logger.info("Quality check | total rows: %s", f"{total:,}")

    # --- null rates (single pass) ---
    present_null_cols = [c for c in _NULL_RATE_COLS if c in flights_df.columns]
    for missing in sorted(set(_NULL_RATE_COLS) - set(present_null_cols)):
        logger.warning("Quality check | column %s not found; skipping null rate", missing)

    if present_null_cols:
        null_row = flights_df.agg(
            *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in present_null_cols]
        ).first()
        for c in present_null_cols:
            nulls = null_row[c] or 0
            rate = (nulls / total) if total else 0.0
            logger.info(
                "Quality check | null rate %s: %.4f%% (%s of %s)",
                c, rate * 100, f"{nulls:,}", f"{total:,}",
            )

    # --- delay distribution ---
    for c in _DELAY_COLS:
        if c not in flights_df.columns:
            logger.warning("Quality check | column %s not found; skipping distribution", c)
            continue
        stats = flights_df.agg(
            F.mean(c).alias("mean"),
            F.stddev(c).alias("stddev"),
            F.min(c).alias("min"),
            F.max(c).alias("max"),
        ).first()
        p50, p95 = flights_df.approxQuantile(c, [0.5, 0.95], 0.001) or (None, None)

        def _fmt(v):
            return "n/a" if v is None else f"{v:.2f}"

        logger.info(
            "Quality check | %s distribution: mean=%s stddev=%s min=%s max=%s p50=%s p95=%s",
            c, _fmt(stats["mean"]), _fmt(stats["stddev"]), _fmt(stats["min"]),
            _fmt(stats["max"]), _fmt(p50), _fmt(p95),
        )

    # --- split counts ---
    if "data_split" in flights_df.columns:
        counts = {
            r["data_split"]: r["count"]
            for r in flights_df.groupBy("data_split").count().collect()
        }
        for split in ("train", "validation", "test"):
            logger.info("Quality check | data_split=%s: %s rows", split, f"{counts.get(split, 0):,}")
        for other in sorted(k for k in counts if k not in ("train", "validation", "test")):
            logger.info("Quality check | data_split=%s: %s rows", other, f"{counts[other]:,}")


# ---------------------------------------------------------------------------
# NOAA weather standardisation
# ---------------------------------------------------------------------------

def standardise_weather(
    noaa_paths: dict,
    year: int = YEAR,
    paths: dict = PATHS,
) -> object:
    """
    Parse all NOAA PSV files, filter to *year*, deduplicate, convert
    timestamps to ISO strings (avoids PyArrow INT64 nanosecond issues),
    persist as a single Parquet file, and return a Spark-ready path string.

    The caller is responsible for reading the Parquet into Spark and parsing
    the 'weather_time_str' column into a proper timestamp.
    """
    weather_parts: list = []
    weather_cols = NOAA_KEEP_COLS  # ['DATE', 'temperature', ...]

    logger.info("Parsing NOAA PSV files…")
    for airport, psv_file in noaa_paths.items():
        pdf = pd.read_csv(psv_file, sep="|", usecols=weather_cols, low_memory=False)
        pdf["airport"] = airport

        # Parse UTC timestamps; drop rows where date is unparseable
        dt_series = pd.to_datetime(pdf["DATE"], errors="coerce", utc=True)
        pdf = pdf.assign(weather_dt=dt_series).dropna(subset=["weather_dt"])

        # Filter to target year
        pdf = pdf[pdf["weather_dt"].dt.year == year]

        # Convert to ISO string — prevents PyArrow from writing INT64(NANOS)
        pdf["weather_time_str"] = pdf["weather_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Coerce all numeric columns
        for col in weather_cols[1:]:  # skip 'DATE'
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

        pdf = (
            pdf
            .drop(columns=["DATE", "weather_dt"])
            .sort_values("weather_time_str")
            .drop_duplicates(subset=["airport", "weather_time_str"], keep="last")
        )
        weather_parts.append(pdf)
        logger.info("Parsed NOAA %s: %s rows", airport, f"{len(pdf):,}")

    combined = pd.concat(weather_parts, ignore_index=True)
    out = paths["processed"] / f"noaa_hourly_{year}.parquet"

    # Remove any stale Parquet directory left by a previous Spark write
    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()

    combined.to_parquet(out, index=False, compression="snappy")
    logger.info("Canonical weather table: %s rows → %s", f"{len(combined):,}", out)
    return out


# ---------------------------------------------------------------------------
# NOAA Storm Events standardisation
# ---------------------------------------------------------------------------

def standardise_storm_events(
    storm_csv_path,
    year: int = YEAR,
    paths: dict = PATHS,
) -> object:
    """
    Parse the NOAA Storm Events detail CSV, assign each event to the nearest
    airport (within STORM_RADIUS_DEG), and persist a compact Parquet with
    columns: airport, event_date, event_hour_utc, event_type.

    Returns the output Path, or None if *storm_csv_path* is None / missing.
    """
    from config.settings import AIRPORT_COORDS, STORM_RADIUS_DEG

    if storm_csv_path is None or not Path(storm_csv_path).exists():
        logger.info("No storm events CSV — storm features will be zero.")
        return None

    logger.info("Parsing NOAA Storm Events…")
    df = pd.read_csv(storm_csv_path, dtype=str, low_memory=False)

    # Parse coordinates
    for c in ("BEGIN_LAT", "BEGIN_LON"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])

    # Parse event datetime
    df["event_dt"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["event_dt"])
    df = df[df["event_dt"].dt.year == year]

    # Match each event to the nearest airport within radius
    rows = []
    for _, row in df.iterrows():
        lat, lon = row["BEGIN_LAT"], row["BEGIN_LON"]
        for arpt, (alat, alon) in AIRPORT_COORDS.items():
            if abs(lat - alat) < STORM_RADIUS_DEG and abs(lon - alon) < STORM_RADIUS_DEG:
                rows.append({
                    "airport": arpt,
                    "event_date": row["event_dt"].strftime("%Y-%m-%d"),
                    "event_hour_utc": row["event_dt"].hour,
                    "event_type": row.get("EVENT_TYPE", "Unknown"),
                })
                break  # assign to first matching airport only

    if not rows:
        logger.warning("No storm events matched our airport cohort.")
        return None

    out_df = pd.DataFrame(rows)
    out = paths["processed"] / f"storm_events_{year}.parquet"
    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()

    out_df.to_parquet(out, index=False, compression="snappy")
    logger.info(
        "Storm events table: %s events at %d airports → %s",
        f"{len(out_df):,}", out_df["airport"].nunique(), out,
    )
    return out


# ---------------------------------------------------------------------------
# Convenience: run all preprocessing steps
# ---------------------------------------------------------------------------

def run_preprocessing(
    spark: SparkSession,
    bts_csv_files: list,
    noaa_paths: dict,
    storm_csv_path=None,
    paths: dict = PATHS,
) -> tuple:
    """
    Run BTS cleaning, data-quality checks, NOAA standardisation, and
    storm events standardisation.
    Returns (flights_df, weather_parquet_path).
    """
    logger.info("=== Flight Cleaning ===")
    flights_df = clean_flights(spark, bts_csv_files, paths=paths)
    run_quality_checks(flights_df)

    logger.info("=== Weather Standardisation ===")
    weather_path = standardise_weather(noaa_paths, paths=paths)

    logger.info("=== Storm Events Standardisation ===")
    standardise_storm_events(storm_csv_path, paths=paths)

    return flights_df, weather_path

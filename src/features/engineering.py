"""
src/features/engineering.py
============================
Derives all predictive features and writes partitioned Parquet feature sets.

Steps
-----
1. UTC conversion of scheduled departure times (local timezone → UTC)
2. Rolling 2-hour airport departure congestion window
3. Prior-leg aircraft rotation delay (lag over tail number)
4. As-of weather join [T-8h, T-2h]
5. Aviation feature decompositions (crosswind, headwind, icing, IFR)
6. Two-stage hurdle labels (DEP_DEL15 / ARR_DEL15, log transforms)
7. Chronological data-split column
8. Persist four partitioned feature sets

Public API
----------
  build_features(spark, flights_df, weather_parquet_path, paths, ...) -> DataFrame
  save_feature_sets(features_df, paths, ...) -> None
  run_feature_engineering(spark, flights_df, weather_parquet_path, paths) -> DataFrame
"""

from pathlib import Path

import airportsdata
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from config.settings import (
    AIRPORT_RUNWAYS,
    BTS_FEATURE_COLS,
    FULL_FEATURE_COLS,
    MIN_WEATHER_COVERAGE,
    PATHS,
    SPLIT_TEST_MONTHS,
    SPLIT_TRAIN_MONTHS,
    SPLIT_VALIDATION_MONTHS,
    TOP_10_AIRPORTS,
    WEATHER_FEATURE_COLS,
    YEAR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# UTC conversion + congestion + rotation delay
# ---------------------------------------------------------------------------

def _add_operational_features(spark: SparkSession, flights) -> object:
    """
    Attach UTC departure timestamps, 2-hour congestion window count, and
    prior-leg rotation delay to *flights*.
    """
    # -- timezone lookup --
    airport_db = airportsdata.load("IATA")
    airport_tz = {c: airport_db[c]["tz"] for c in TOP_10_AIRPORTS}

    tz_schema = T.StructType([
        T.StructField("Origin",    T.StringType(), False),
        T.StructField("origin_tz", T.StringType(), False),
    ])
    tz_df = spark.createDataFrame(list(airport_tz.items()), schema=tz_schema)
    flights = flights.join(tz_df, on="Origin", how="left")

    # -- local scheduled departure → UTC --
    flights = (
        flights
        .withColumn("dep_hh", F.floor(F.col("CRSDepTime") / 100).cast("int"))
        .withColumn("dep_mm", (F.col("CRSDepTime") % 100).cast("int"))
        .withColumn(
            "sched_dep_local",
            F.to_timestamp(
                F.concat_ws(
                    " ",
                    F.col("FlightDate").cast("string"),
                    F.format_string(
                        "%02d:%02d:00",
                        F.when(F.col("dep_hh") == 24, 0).otherwise(F.col("dep_hh")),
                        F.col("dep_mm"),
                    ),
                )
            ),
        )
        # Handle midnight edge case: CRSDepTime == 2400
        .withColumn(
            "sched_dep_local",
            F.when(
                F.col("CRSDepTime") == 2400,
                F.expr("sched_dep_local + INTERVAL 1 DAY"),
            ).otherwise(F.col("sched_dep_local")),
        )
        .withColumn(
            "scheduled_departure_utc",
            F.to_utc_timestamp("sched_dep_local", F.col("origin_tz")),
        )
        .withColumn("t_minus_2_utc", F.expr("scheduled_departure_utc - INTERVAL 2 HOURS"))
        .withColumn("flight_key", F.monotonically_increasing_id())
    )

    # -- Feature 1: rolling 2-hour departure congestion --
    time_sec = F.col("scheduled_departure_utc").cast("long")
    cong_win = Window.partitionBy("Origin").orderBy(time_sec).rangeBetween(-7200, 0)
    flights = flights.withColumn(
        "origin_congestion_2h", F.count("flight_key").over(cong_win) - 1
    )

    # -- Feature 2: prior-leg aircraft rotation delay --
    if "Tail_Number" in flights.columns:
        tail_win = Window.partitionBy("Tail_Number").orderBy("scheduled_departure_utc")
        flights = (
            flights
            .withColumn("prev_arr_delay",      F.lag("ArrDelay",              1).over(tail_win))
            .withColumn("prev_dep_utc",        F.lag("scheduled_departure_utc", 1).over(tail_win))
            .withColumn("prev_elapsed_min",    F.lag("CRSElapsedTime",          1).over(tail_win))
            # Estimated scheduled arrival of the previous leg
            .withColumn(
                "prev_sched_arr_utc",
                F.expr("prev_dep_utc + make_interval(0, 0, 0, 0, 0, cast(prev_elapsed_min as int), 0)"),
            )
            .withColumn(
                "prior_leg_delay",
                F.when(
                    # Previous leg's scheduled arrival must be known by T-2h
                    F.col("prev_sched_arr_utc").isNotNull() &
                    (F.col("prev_sched_arr_utc") <= F.col("t_minus_2_utc")) &
                    # Must be a strictly earlier flight (not the same flight)
                    (F.col("prev_dep_utc") < F.col("scheduled_departure_utc")),
                    F.coalesce(F.col("prev_arr_delay"), F.lit(0.0)),
                ).otherwise(0.0),
            )
            .drop("prev_sched_arr_utc", "prev_elapsed_min")
        )
    else:
        flights = flights.withColumn("prior_leg_delay", F.lit(0.0))

    return flights


# ---------------------------------------------------------------------------
# As-of weather join
# ---------------------------------------------------------------------------

def _join_weather(spark: SparkSession, flights, weather_parquet_path: Path) -> object:
    """
    Left-join each flight with the most recent weather observation in the
    window [T-8h, T-2h], where T is the scheduled UTC departure.
    """
    weather_spark = (
        spark.read.parquet(str(weather_parquet_path))
        .withColumn(
            "weather_time_utc",
            F.to_timestamp("weather_time_str", "yyyy-MM-dd HH:mm:ss"),
        )
        .drop("weather_time_str")
    )

    w_clean = weather_spark.select(
        F.col("airport").alias("w_airport"),
        F.col("weather_time_utc").cast("timestamp").alias("w_time_utc"),
        F.col("temperature").alias("w_temp"),
        F.col("dew_point_temperature").alias("w_dew"),
        F.col("wind_direction").alias("w_wdir"),
        F.col("wind_speed").alias("w_wspeed"),
        F.col("visibility").alias("w_vis"),
        F.col("sea_level_pressure").alias("w_slp"),
    )

    join_cond = (
        (F.col("Origin") == F.col("w_airport")) &
        (F.col("w_time_utc") <= F.col("t_minus_2_utc")) &
        (F.col("w_time_utc") >= F.expr("t_minus_2_utc - INTERVAL 6 HOURS"))
    )

    matched = flights.join(w_clean, join_cond, "left")

    # Keep only the most recent observation per flight
    dedup_win = Window.partitionBy("flight_key").orderBy(
        F.col("w_time_utc").desc_nulls_last()
    )
    joined = (
        matched
        .withColumn("rn", F.row_number().over(dedup_win))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    return joined


# ---------------------------------------------------------------------------
# Weather coverage reporting
# ---------------------------------------------------------------------------

def _add_weather_available(flights) -> object:
    """Flag flights that matched a weather observation (1) or not (0)."""
    return flights.withColumn(
        "weather_available", F.col("w_time_utc").isNotNull().cast("int")
    )


def _report_weather_coverage(flights, min_coverage: float = MIN_WEATHER_COVERAGE) -> float:
    """
    Log overall and per-airport weather coverage after the join.
    Emits a WARNING when overall coverage is below *min_coverage*.
    Returns the overall coverage fraction.
    """
    per_airport = (
        flights.groupBy("Origin")
        .agg(
            F.count(F.lit(1)).alias("total"),
            F.sum("weather_available").alias("with_weather"),
        )
        .orderBy("Origin")
        .collect()
    )

    total = int(sum(r["total"] for r in per_airport))
    with_weather = int(sum(r["with_weather"] or 0 for r in per_airport))
    coverage = (with_weather / total) if total else 0.0

    logger.info("Weather join: total flights joined = %d", total)
    logger.info("Weather join: flights with weather_available == 1 = %d", with_weather)
    logger.info("Weather join: coverage = %.2f%%", coverage * 100.0)
    for r in per_airport:
        n = int(r["total"])
        w = int(r["with_weather"] or 0)
        logger.info(
            "Weather coverage [%s]: %d / %d (%.2f%%)",
            r["Origin"], w, n, (w / n * 100.0) if n else 0.0,
        )

    if coverage < min_coverage:
        logger.warning(
            "Weather coverage %.2f%% is below the minimum %.2f%%; "
            "weather features may be unreliable.",
            coverage * 100.0, min_coverage * 100.0,
        )
    return coverage


# ---------------------------------------------------------------------------
# BTS delay-cause daily airport aggregates (same-day, prior-flight)
# ---------------------------------------------------------------------------

def _add_delay_cause_aggregates(flights) -> object:
    """
    For each flight, compute same-day prior-flight delay-cause aggregates
    at the origin airport.

    Leakage fix: only flights whose scheduled departure is at or before
    t_minus_2_utc (i.e. known at prediction time T-2h) are included.

    Implementation: use rangeBetween with an upper bound of -7200 seconds
    (2 hours before the current row's departure time) on a window ordered by
    epoch-seconds. This correctly excludes the current row AND any flight
    that departed within the 2-hour prediction horizon.
    """
    dep_sec = F.col("scheduled_departure_utc").cast("long")

    # Window: same Origin, same UTC calendar date, ordered by epoch-second.
    # Upper bound of -7200 means: include only flights whose scheduled
    # departure is at least 2 hours before the current flight's departure.
    # These are the flights whose outcomes would be known at T-2h.
    day_win = (
        Window.partitionBy("Origin", F.to_date("scheduled_departure_utc"))
        .orderBy(dep_sec)
        .rangeBetween(Window.unboundedPreceding, -7200)
    )

    result = (
        flights
        # NAS delays
        .withColumn("_nas_flag",   F.when(F.col("NASDelay") > 0, 1).otherwise(0))
        .withColumn("airport_nas_delays_today",
                    F.coalesce(F.sum("_nas_flag").over(day_win), F.lit(0)).cast("double"))
        .withColumn("airport_nas_delay_min_today",
                    F.coalesce(F.sum("NASDelay").over(day_win), F.lit(0.0)))
        # Weather delays
        .withColumn("_wx_flag",    F.when(F.col("WeatherDelay") > 0, 1).otherwise(0))
        .withColumn("airport_wx_delays_today",
                    F.coalesce(F.sum("_wx_flag").over(day_win), F.lit(0)).cast("double"))
        .withColumn("airport_wx_delay_min_today",
                    F.coalesce(F.sum("WeatherDelay").over(day_win), F.lit(0.0)))
        # Carrier delays
        .withColumn("_cr_flag",    F.when(F.col("CarrierDelay") > 0, 1).otherwise(0))
        .withColumn("airport_carrier_delays_today",
                    F.coalesce(F.sum("_cr_flag").over(day_win), F.lit(0)).cast("double"))
        # Late-aircraft cascade
        .withColumn("_la_flag",    F.when(F.col("LateAircraftDelay") > 0, 1).otherwise(0))
        .withColumn("airport_late_acft_delays_today",
                    F.coalesce(F.sum("_la_flag").over(day_win), F.lit(0)).cast("double"))
        .withColumn("airport_late_acft_min_today",
                    F.coalesce(F.sum("LateAircraftDelay").over(day_win), F.lit(0.0)))
        # Overall delay picture
        .withColumn("_del15_flag", F.when(F.col("DepDelay") > 15, 1).otherwise(0))
        .withColumn("airport_total_delayed_today",
                    F.coalesce(F.sum("_del15_flag").over(day_win), F.lit(0)).cast("double"))
        .withColumn("_prior_count",
                    F.coalesce(F.count("flight_key").over(day_win), F.lit(0)))
        .withColumn("airport_avg_delay_today",
                    F.coalesce(
                        F.when(
                            F.col("_prior_count") > 0,
                            F.sum("DepDelay").over(day_win) / F.col("_prior_count"),
                        ),
                        F.lit(0.0),
                    ))
        .withColumn("airport_delay_rate_today",
                    F.when(F.col("_prior_count") > 0,
                           F.col("airport_total_delayed_today") / F.col("_prior_count"))
                    .otherwise(0.0))
        .drop("_nas_flag", "_wx_flag", "_cr_flag", "_la_flag",
              "_del15_flag", "_prior_count")
    )
    logger.info(
        "Added 10 delay-cause daily aggregate features "
        "(rangeBetween T-2h cap — no leakage)"
    )
    return result


# ---------------------------------------------------------------------------
# Storm event counts + holiday flags
# ---------------------------------------------------------------------------

def _add_contextual_features(spark: SparkSession, flights, paths: dict = PATHS) -> object:
    """
    Add storm-event counts (6h / 24h windows) and holiday flags.

    Storm events: joined from the preprocessed storm_events Parquet.
    Holidays: loaded from the holidays JSON and broadcast-joined.
    Both gracefully fall back to zeros if the data files are missing.
    """
    # ── Storm events ─────────────────────────────────────────────────────
    storm_path = paths["processed"] / f"storm_events_{YEAR}.parquet"
    if storm_path.exists():
        logger.info("Joining storm event counts")
        storms = spark.read.parquet(str(storm_path))

        # Build a per-airport, per-date count
        storm_daily = (
            storms
            .withColumn("event_date", F.to_date("event_date"))
            .groupBy("airport", "event_date")
            .agg(
                F.count("*").alias("_storm_count_24h"),
                F.sum(
                    F.when(F.col("event_hour_utc") >= 6, 1).otherwise(0)
                ).alias("_storm_count_6h"),  # daytime events (06-23 UTC)
            )
        )

        flights = flights.withColumn("_fdate", F.to_date("FlightDate"))
        flights = flights.join(
            storm_daily,
            on=(
                (F.col("Origin") == F.col("airport")) &
                (F.col("_fdate") == F.col("event_date"))
            ),
            how="left",
        )
        flights = (
            flights
            .withColumn("storm_events_24h",
                        F.coalesce(F.col("_storm_count_24h"), F.lit(0)).cast("double"))
            .withColumn("storm_events_6h",
                        F.coalesce(F.col("_storm_count_6h"), F.lit(0)).cast("double"))
            .drop("_fdate", "airport", "event_date", "_storm_count_24h", "_storm_count_6h")
        )
        logger.info("Storm event features added")
    else:
        logger.info("No storm events Parquet — filling storm features with zeros")
        flights = (
            flights
            .withColumn("storm_events_24h", F.lit(0.0))
            .withColumn("storm_events_6h",  F.lit(0.0))
        )

    # ── Holidays ─────────────────────────────────────────────────────────
    holidays_path = paths["processed"] / f"us_holidays_{YEAR}.json"
    if holidays_path.exists():
        import json as _json
        with open(holidays_path) as fh:
            holidays_data = _json.load(fh)
        holiday_dates = {h["date"] for h in holidays_data}

        # Also compute adjacent dates (day before + day after each holiday)
        from datetime import datetime, timedelta
        adjacent_dates = set()
        for d in holiday_dates:
            dt = datetime.strptime(d, "%Y-%m-%d")
            adjacent_dates.add((dt - timedelta(days=1)).strftime("%Y-%m-%d"))
            adjacent_dates.add((dt + timedelta(days=1)).strftime("%Y-%m-%d"))
            adjacent_dates.add(d)

        logger.info("Adding holiday flags (%d holidays, %d adjacent dates)",
                    len(holiday_dates), len(adjacent_dates))

        holiday_list = sorted(holiday_dates)
        adjacent_list = sorted(adjacent_dates)

        flights = (
            flights
            .withColumn("_date_str", F.col("FlightDate").cast("string"))
            .withColumn("is_holiday",
                        F.when(F.col("_date_str").isin(holiday_list), 1.0).otherwise(0.0))
            .withColumn("is_holiday_adjacent",
                        F.when(F.col("_date_str").isin(adjacent_list), 1.0).otherwise(0.0))
            .drop("_date_str")
        )
    else:
        logger.info("No holidays JSON — filling holiday features with zeros")
        flights = (
            flights
            .withColumn("is_holiday",          F.lit(0.0))
            .withColumn("is_holiday_adjacent", F.lit(0.0))
        )

    return flights


# ---------------------------------------------------------------------------
# Aviation feature derivation + labels + split
# ---------------------------------------------------------------------------

def _add_derived_features(flights) -> object:
    """
    Add crosswind/headwind decomposition, icing risk, IFR flag,
    hurdle labels, and chronological data-split column.
    """
    runway_map = F.create_map(
        [F.lit(x) for kv in AIRPORT_RUNWAYS.items() for x in kv]
    )
    rad = np.pi / 180.0
    wind_diff = (F.col("w_wdir") - runway_map[F.col("Origin")]) * F.lit(rad)

    train_lo, train_hi   = SPLIT_TRAIN_MONTHS
    val_lo,   val_hi     = SPLIT_VALIDATION_MONTHS
    test_lo,  test_hi    = SPLIT_TEST_MONTHS

    return (
        flights
        .withColumn("scheduled_departure_hour", F.hour("sched_dep_local").cast("double"))
        .withColumn("weather_available",   F.col("w_time_utc").isNotNull().cast("int"))
        .withColumn(
            "weather_age_minutes",
            F.coalesce(
                (F.col("t_minus_2_utc").cast("long") - F.col("w_time_utc").cast("long")) / 60.0,
                F.lit(0.0),   # no weather observation → age is unknown; 0 with weather_available=0
            ),
        )
        .withColumn("weather_crosswind",   F.abs(F.col("w_wspeed") * F.sin(wind_diff)))
        .withColumn("weather_headwind",    F.col("w_wspeed") * F.cos(wind_diff))
        .withColumn(
            "weather_icing_risk",
            F.when(
                (F.col("w_temp") <= 0.0) & ((F.col("w_temp") - F.col("w_dew")) <= 2.0),
                1.0,
            ).otherwise(0.0),
        )
        .withColumn(
            "weather_ifr_conditions",
            F.when(F.col("w_vis") < 4800.0, 1.0).otherwise(0.0),
        )
        # Hurdle labels
        .withColumn("DEP_DEL15",     F.when(F.col("DepDelay") >= 15.0, 1.0).otherwise(0.0))
        .withColumn("ARR_DEL15",     F.when(F.col("ArrDelay") >= 15.0, 1.0).otherwise(0.0))
        .withColumn(
            "log_DepDelay",
            F.log1p(F.when(F.col("DepDelay") > 0, F.col("DepDelay")).otherwise(0.0)),
        )
        .withColumn(
            "log_ArrDelay",
            F.log1p(F.when(F.col("ArrDelay") > 0, F.col("ArrDelay")).otherwise(0.0)),
        )
        # Chronological split
        .withColumn(
            "data_split",
            F.when(F.month("FlightDate").between(train_lo, train_hi), "train")
             .when(F.month("FlightDate").isin(*range(val_lo, val_hi + 1)), "validation")
             .when(F.month("FlightDate").isin(*range(test_lo, test_hi + 1)), "test"),
        )
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_features(
    spark: SparkSession,
    flights_df,
    weather_parquet_path: Path,
    paths: dict = PATHS,
) -> object:
    """
    Run all feature-engineering steps and return the enriched DataFrame.
    Does NOT write to disk — call save_feature_sets() separately.
    """
    logger.info("Adding operational features (UTC, congestion, rotation delay)")
    df = _add_operational_features(spark, flights_df)

    logger.info("Joining weather observations")
    df = _join_weather(spark, df, weather_parquet_path)
    df = _add_weather_available(df)
    _report_weather_coverage(df)

    logger.info("Deriving aviation features and labels")
    df = _add_derived_features(df)

    logger.info("Computing delay-cause daily aggregates")
    df = _add_delay_cause_aggregates(df)

    logger.info("Adding contextual features (storm events + holidays)")
    df = _add_contextual_features(spark, df, paths=paths)

    return df


def save_feature_sets(
    features_df,
    paths: dict = PATHS,
    bts_feature_cols: list = BTS_FEATURE_COLS,
    weather_feature_cols: list = WEATHER_FEATURE_COLS,
    full_feature_cols: list = FULL_FEATURE_COLS,
) -> None:
    """
    Write six partitioned Parquet feature sets:
      departure_bts_only / departure_bts_weather / departure_full
      arrival_bts_only   / arrival_bts_weather   / arrival_full
    """
    for task, raw_tgt, log_tgt, cls_tgt in [
        ("departure", "DepDelay", "log_DepDelay", "DEP_DEL15"),
        ("arrival",   "ArrDelay", "log_ArrDelay", "ARR_DEL15"),
    ]:
        for variant, p_cols in [
            ("bts_only",    bts_feature_cols),
            ("bts_weather", weather_feature_cols),
            ("full",        full_feature_cols),
        ]:
            out_dir = paths["features"] / f"{task}_{variant}"
            meta = ["flight_key", "FlightDate", "data_split", raw_tgt, log_tgt, cls_tgt]
            ds = features_df.select(
                *list(dict.fromkeys(meta + p_cols))
            ).filter(F.col("data_split").isNotNull())

            (
                ds.write
                .mode("overwrite")
                .partitionBy("data_split")
                .option("compression", "snappy")
                .parquet(str(out_dir))
            )
            logger.info("Saved feature set: %s", out_dir)


def run_feature_engineering(
    spark: SparkSession,
    flights_df,
    weather_parquet_path: Path,
    paths: dict = PATHS,
) -> object:
    """
    End-to-end feature engineering: build + save.
    Returns the enriched features DataFrame.
    """
    logger.info("=== Feature Engineering ===")
    features_df = build_features(spark, flights_df, weather_parquet_path, paths=paths)

    # Materialise once so save_feature_sets doesn't re-execute the full DAG
    # for each of the 6 feature-set writes.
    features_df = features_df.cache()
    cached_count = features_df.count()
    logger.info("Features cached: %s rows", f"{cached_count:,}")

    save_feature_sets(features_df, paths=paths)
    features_df.unpersist()
    return features_df

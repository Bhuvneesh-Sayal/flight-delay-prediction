"""
src/streaming/pipeline.py
==========================
Spark Structured Streaming pipeline for low-latency flight delay scoring.

Architecture
------------

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │  Flight events  │────▶│  Watermark join  │────▶│  Hurdle model score  │
  │  (file / Kafka) │     │  with weather    │     │  (batch micro-batch) │
  └─────────────────┘     └──────────────────┘     └──────────────────────┘
                                  │
                          15-min watermark on
                          scheduled_departure_utc

Design decisions
----------------
* Source format: JSON (file source for local dev; swap to Kafka for prod by
  changing `_make_flight_source` / `_make_weather_source`).
* The hurdle pipeline model is loaded once at startup and broadcast to every
  micro-batch via a closure — no repeated I/O per batch.
* Weather is treated as a slowly-changing reference table: loaded as a static
  DataFrame and joined inside each micro-batch via `foreachBatch`.
* 15-minute watermark on `scheduled_departure_utc` tolerates late-arriving
  events up to 15 minutes after the event time.

Directories
-----------
  data/streaming/input/flights/   ← drop JSON flight event files here
  data/streaming/input/weather/   ← drop JSON weather update files here
  data/streaming/output/scores/   ← scored predictions land here (Parquet)
  data/streaming/checkpoints/     ← Spark streaming checkpoint location

Running
-------
  python -m src.streaming.pipeline          # uses defaults from config
  python main.py --steps stream             # via main.py

Public API
----------
  run_streaming_pipeline(spark, paths, model_path, timeout_secs) -> None
"""

from pathlib import Path

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession

from config.settings import PATHS, SPARK_CONFIG, WEATHER_FEATURE_COLS, YEAR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

FLIGHT_EVENT_SCHEMA = T.StructType([
    T.StructField("flight_key",              T.LongType(),    True),
    T.StructField("FlightDate",              T.StringType(),  True),   # "YYYY-MM-DD"
    T.StructField("Reporting_Airline",       T.StringType(),  True),
    T.StructField("Origin",                  T.StringType(),  True),
    T.StructField("Dest",                    T.StringType(),  True),
    T.StructField("CRSDepTime",              T.IntegerType(), True),   # e.g. 1430
    T.StructField("CRSElapsedTime",          T.DoubleType(),  True),
    T.StructField("scheduled_departure_utc", T.StringType(),  True),   # ISO-8601
    T.StructField("origin_congestion_2h",    T.LongType(),    True),
    T.StructField("prior_leg_delay",         T.DoubleType(),  True),
    T.StructField("scheduled_month",         T.IntegerType(), True),
    T.StructField("scheduled_day_of_week",   T.IntegerType(), True),
    T.StructField("scheduled_departure_hour",T.DoubleType(),  True),
])

WEATHER_UPDATE_SCHEMA = T.StructType([
    T.StructField("airport",          T.StringType(),  True),
    T.StructField("weather_time_str", T.StringType(),  True),   # "YYYY-MM-DD HH:MM:SS"
    T.StructField("temperature",      T.DoubleType(),  True),
    T.StructField("dew_point_temperature", T.DoubleType(), True),
    T.StructField("wind_direction",   T.DoubleType(),  True),
    T.StructField("wind_speed",       T.DoubleType(),  True),
    T.StructField("visibility",       T.DoubleType(),  True),
    T.StructField("sea_level_pressure", T.DoubleType(), True),
])


# ---------------------------------------------------------------------------
# Source factories  (file source — swap inner function for Kafka in prod)
# ---------------------------------------------------------------------------

def _make_flight_source(spark: SparkSession, input_dir: Path) -> DataFrame:
    """
    Read flight events as a streaming DataFrame from JSON files dropped into
    *input_dir*.  In production replace this with a Kafka readStream.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    return (
        spark.readStream
        .schema(FLIGHT_EVENT_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .json(str(input_dir))
        .withColumn(
            "event_time",
            F.to_timestamp("scheduled_departure_utc").cast("timestamp"),
        )
        .withWatermark("event_time", "15 minutes")
    )


def _make_weather_source(spark: SparkSession, weather_parquet: Path) -> DataFrame:
    """
    Load the canonical weather Parquet as a *static* DataFrame.
    In a production deployment this would be a Delta table or a Kafka
    compacted topic read via readStream.
    """
    return (
        spark.read.parquet(str(weather_parquet))
        .withColumn(
            "w_time_utc",
            F.to_timestamp("weather_time_str", "yyyy-MM-dd HH:mm:ss"),
        )
        .drop("weather_time_str")
        .withColumnRenamed("airport", "w_airport")
        .withColumnRenamed("temperature",           "w_temp")
        .withColumnRenamed("dew_point_temperature", "w_dew")
        .withColumnRenamed("wind_direction",        "w_wdir")
        .withColumnRenamed("wind_speed",            "w_wspeed")
        .withColumnRenamed("visibility",            "w_vis")
        .withColumnRenamed("sea_level_pressure",    "w_slp")
    )


# ---------------------------------------------------------------------------
# Per-micro-batch processing
# ---------------------------------------------------------------------------

def _make_batch_processor(
    weather_df: DataFrame,
    hurdle_model: PipelineModel,
    output_dir: Path,
):
    """
    Return a closure suitable for ``writeStream.foreachBatch``.

    Closure captures the static weather DataFrame and the loaded hurdle model
    so they are not reloaded on every micro-batch.
    """
    import numpy as np
    from pyspark.sql.window import Window

    AIRPORT_RUNWAYS = {
        "ATL":  90.0, "CLT": 180.0, "DEN": 170.0, "DFW": 175.0,
        "LAS": 260.0, "LAX": 250.0, "MCO": 180.0, "ORD": 270.0,
        "PHX": 260.0, "SEA": 160.0,
    }
    runway_map = F.create_map([F.lit(x) for kv in AIRPORT_RUNWAYS.items() for x in kv])
    rad = np.pi / 180.0

    get_prob1 = F.udf(lambda v: float(v[1]), T.DoubleType())

    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            logger.debug("Batch %d is empty — skipping.", batch_id)
            return

        logger.info("Processing streaming batch %d (%d rows)", batch_id, batch_df.count())

        # 1. As-of weather join  [T-6h, T-2h]
        t_minus_2 = F.col("event_time") - F.expr("INTERVAL 2 HOURS")
        t_minus_8 = F.col("event_time") - F.expr("INTERVAL 8 HOURS")

        join_cond = (
            (F.col("Origin") == F.col("w_airport")) &
            (F.col("w_time_utc") <= t_minus_2) &
            (F.col("w_time_utc") >= t_minus_8)
        )
        matched = batch_df.join(weather_df, join_cond, "left")
        dedup_win = Window.partitionBy("flight_key").orderBy(F.col("w_time_utc").desc_nulls_last())
        joined = (
            matched
            .withColumn("rn", F.row_number().over(dedup_win))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )

        # 2. Aviation weather features
        wind_diff = (F.col("w_wdir") - runway_map[F.col("Origin")]) * F.lit(rad)
        enriched = (
            joined
            .withColumn("weather_available",    F.col("w_time_utc").isNotNull().cast("int"))
            .withColumn("weather_age_minutes",
                (t_minus_2.cast("long") - F.col("w_time_utc").cast("long")) / 60.0)
            .withColumn("weather_crosswind",    F.abs(F.col("w_wspeed") * F.sin(wind_diff)))
            .withColumn("weather_headwind",     F.col("w_wspeed") * F.cos(wind_diff))
            .withColumn("weather_icing_risk",
                F.when((F.col("w_temp") <= 0.0) &
                       ((F.col("w_temp") - F.col("w_dew")) <= 2.0), 1.0).otherwise(0.0))
            .withColumn("weather_ifr_conditions",
                F.when(F.col("w_vis") < 4800.0, 1.0).otherwise(0.0))
        )

        # 3. Run hurdle model — classifier scores
        scored = hurdle_model.transform(enriched)
        output = (
            scored
            .withColumn("p_delayed", get_prob1("probability"))
            .select(
                "flight_key",
                "FlightDate",
                "Origin",
                "Dest",
                "Reporting_Airline",
                "scheduled_departure_utc",
                "p_delayed",
                F.current_timestamp().alias("scored_at"),
            )
        )

        output.write.mode("append").parquet(str(output_dir))
        logger.info("  Batch %d: scored %d flights.", batch_id, output.count())

    return process_batch


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_streaming_pipeline(
    spark: SparkSession,
    paths: dict = PATHS,
    task: str = "departure",
    timeout_secs: int = 60,
) -> None:
    """
    Start the Structured Streaming pipeline.

    Parameters
    ----------
    spark        : active SparkSession
    paths        : project paths dict
    task         : 'departure' or 'arrival'
    timeout_secs : how long to run before stopping (useful for testing).
                   Set to -1 to run until manually interrupted.
    """
    # ── directories ─────────────────────────────────────────────────────────
    stream_base   = paths.get("streaming_input",  PATHS["interim"].parent / "streaming")
    flight_input  = stream_base / "input"  / "flights"
    output_dir    = stream_base / "output" / "scores"
    checkpoint    = stream_base / "checkpoints"
    for d in [flight_input, output_dir, checkpoint]:
        d.mkdir(parents=True, exist_ok=True)

    weather_parquet = paths["processed"] / f"noaa_hourly_{YEAR}.parquet"

    # Try hurdle model paths in order of preference (matches hurdle.py save pattern)
    clf_model_dir = paths["models"] / f"hurdle_{task}_full_classifier"
    if not clf_model_dir.exists():
        clf_model_dir = paths["models"] / f"hurdle_{task}_bts_weather_classifier"

    if not weather_parquet.exists():
        raise FileNotFoundError(
            f"Weather Parquet not found at {weather_parquet}. Run 'prep' step first."
        )
    if not clf_model_dir.exists():
        raise FileNotFoundError(
            f"Hurdle classifier not found at {clf_model_dir}. Run 'hurdle' step first."
        )

    logger.info("Loading static weather reference…")
    weather_df = _make_weather_source(spark, weather_parquet)
    weather_df.cache()

    logger.info("Loading hurdle classifier from %s…", clf_model_dir)
    hurdle_model = PipelineModel.load(str(clf_model_dir))

    logger.info("Starting streaming pipeline (input: %s)…", flight_input)
    flight_stream = _make_flight_source(spark, flight_input)

    batch_fn = _make_batch_processor(weather_df, hurdle_model, output_dir)

    query = (
        flight_stream.writeStream
        .foreachBatch(batch_fn)
        .option("checkpointLocation", str(checkpoint))
        .trigger(processingTime="30 seconds")
        .start()
    )

    logger.info(
        "Streaming query started (id=%s). "
        "Drop JSON files into %s to trigger scoring.",
        query.id, flight_input,
    )

    if timeout_secs > 0:
        query.awaitTermination(timeout=timeout_secs)
        query.stop()
        logger.info("Streaming pipeline stopped after %ds timeout.", timeout_secs)
    else:
        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            query.stop()
            logger.info("Streaming pipeline stopped by user.")

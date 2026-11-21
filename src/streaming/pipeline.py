"""
src/streaming/pipeline.py
==========================
Spark Structured Streaming pipeline for low-latency flight delay scoring.

Architecture
------------

  ┌──────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
  │  Flight events       │────▶│  Weather join    │────▶│  Hurdle model score │
  │  Kafka / file source │     │  (static NOAA)   │     │  per micro-batch    │
  └──────────────────────┘     └──────────────────┘     └─────────────────────┘
                                        │                         │
                                15-min watermark          Kafka output topic
                                on event_time             delay-predictions
                                                      + Parquet backup in
                                                        data/streaming/output/

Sources
-------
  "kafka"  — reads from KAFKA_FLIGHT_TOPIC via the Spark SQL Kafka connector.
             Requires spark.jars.packages = KAFKA_SPARK_PACKAGE in SparkSession.
  "file"   — reads JSON files dropped into STREAMING_INPUT_DIR/flights/.
             Useful for local testing without Kafka running.

Running
-------
  python main.py --steps stream              # file source (default)
  python main.py --steps stream --kafka      # Kafka source

Public API
----------
  run_streaming_pipeline(spark, paths, task, source, timeout_secs) -> None
"""

from pathlib import Path

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession

from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_FLIGHT_TOPIC,
    KAFKA_SCORES_TOPIC,
    PATHS,
    STREAMING_CHECKPOINT,
    STREAMING_INPUT_DIR,
    STREAMING_OUTPUT_DIR,
    STREAMING_TRIGGER_SECONDS,
    YEAR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema for a flight event message
# (must match what the producer serialises and what engineering.py produces)
# ---------------------------------------------------------------------------
FLIGHT_EVENT_SCHEMA = T.StructType([
    T.StructField("flight_key",               T.LongType(),    True),
    T.StructField("FlightDate",               T.StringType(),  True),
    T.StructField("Reporting_Airline",        T.StringType(),  True),
    T.StructField("Origin",                   T.StringType(),  True),
    T.StructField("Dest",                     T.StringType(),  True),
    T.StructField("CRSDepTime",               T.IntegerType(), True),
    T.StructField("CRSElapsedTime",           T.DoubleType(),  True),
    T.StructField("scheduled_departure_utc",  T.StringType(),  True),  # ISO-8601
    T.StructField("origin_congestion_2h",     T.LongType(),    True),
    T.StructField("prior_leg_delay",          T.DoubleType(),  True),
    T.StructField("scheduled_month",          T.IntegerType(), True),
    T.StructField("scheduled_day_of_week",    T.IntegerType(), True),
    T.StructField("scheduled_departure_hour", T.DoubleType(),  True),
])

# Schema for scored output written to the Kafka scores topic
SCORE_SCHEMA = T.StructType([
    T.StructField("flight_key",              T.LongType(),    True),
    T.StructField("FlightDate",              T.StringType(),  True),
    T.StructField("Origin",                  T.StringType(),  True),
    T.StructField("Dest",                    T.StringType(),  True),
    T.StructField("Reporting_Airline",       T.StringType(),  True),
    T.StructField("scheduled_departure_utc", T.StringType(),  True),
    T.StructField("p_delayed",               T.DoubleType(),  True),
    T.StructField("scored_at",               T.StringType(),  True),
])


# ---------------------------------------------------------------------------
# Source factories
# ---------------------------------------------------------------------------

def _make_kafka_flight_source(spark: SparkSession) -> DataFrame:
    """
    Read flight events from the Kafka KAFKA_FLIGHT_TOPIC.

    Each Kafka message value is expected to be a JSON string matching
    FLIGHT_EVENT_SCHEMA.  The watermark is applied on event_time
    (parsed from scheduled_departure_utc).
    """
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_FLIGHT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    return (
        raw
        # Kafka delivers bytes; cast value to string then parse JSON
        .select(F.from_json(F.col("value").cast("string"), FLIGHT_EVENT_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_time", F.to_timestamp("scheduled_departure_utc").cast("timestamp"))
        .withWatermark("event_time", "15 minutes")
    )


def _make_file_flight_source(spark: SparkSession, input_dir: Path) -> DataFrame:
    """
    Read flight events from JSON files dropped into *input_dir*.
    Used for local testing without Kafka running.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    return (
        spark.readStream
        .schema(FLIGHT_EVENT_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .json(str(input_dir))
        .withColumn("event_time", F.to_timestamp("scheduled_departure_utc").cast("timestamp"))
        .withWatermark("event_time", "15 minutes")
    )


def _make_weather_reference(spark: SparkSession, weather_parquet: Path) -> DataFrame:
    """Load static NOAA weather Parquet as a broadcast-eligible reference table."""
    return (
        spark.read.parquet(str(weather_parquet))
        .withColumn("w_time_utc",
                    F.to_timestamp("weather_time_str", "yyyy-MM-dd HH:mm:ss"))
        .drop("weather_time_str")
        .withColumnRenamed("airport",               "w_airport")
        .withColumnRenamed("temperature",           "w_temp")
        .withColumnRenamed("dew_point_temperature", "w_dew")
        .withColumnRenamed("wind_direction",        "w_wdir")
        .withColumnRenamed("wind_speed",            "w_wspeed")
        .withColumnRenamed("visibility",            "w_vis")
        .withColumnRenamed("sea_level_pressure",    "w_slp")
    )


# ---------------------------------------------------------------------------
# Per-micro-batch processor
# ---------------------------------------------------------------------------

def _make_batch_processor(
    weather_df: DataFrame,
    hurdle_model: PipelineModel,
    output_dir: Path,
    write_to_kafka: bool,
    spark: SparkSession,
):
    """
    Return a closure for ``writeStream.foreachBatch``.

    The closure captures the static weather DataFrame and the loaded hurdle
    model so they are not reloaded on every micro-batch.

    Each batch:
    1. Performs an as-of weather join [T-8h, T-2h]
    2. Derives aviation features (crosswind, headwind, icing, IFR)
    3. Scores with the hurdle classifier
    4. Writes predictions to:
       - Kafka scores topic (if write_to_kafka=True)
       - Parquet backup under output_dir (always)
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

        n_in = batch_df.count()
        logger.info("Batch %d: %d events received.", batch_id, n_in)

        # ── 1. As-of weather join [T-8h, T-2h] ──────────────────────────
        t_minus_2 = F.col("event_time") - F.expr("INTERVAL 2 HOURS")
        t_minus_8 = F.col("event_time") - F.expr("INTERVAL 8 HOURS")

        join_cond = (
            (F.col("Origin")     == F.col("w_airport")) &
            (F.col("w_time_utc") <= t_minus_2) &
            (F.col("w_time_utc") >= t_minus_8)
        )
        matched = batch_df.join(weather_df, join_cond, "left")

        dedup_win = (
            Window.partitionBy("flight_key")
            .orderBy(F.col("w_time_utc").desc_nulls_last())
        )
        joined = (
            matched
            .withColumn("rn", F.row_number().over(dedup_win))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )

        # ── 2. Aviation weather features ─────────────────────────────────
        wind_diff = (F.col("w_wdir") - runway_map[F.col("Origin")]) * F.lit(rad)
        enriched = (
            joined
            .withColumn("weather_available",
                        F.col("w_time_utc").isNotNull().cast("int"))
            .withColumn("weather_age_minutes",
                        F.coalesce(
                            (t_minus_2.cast("long") - F.col("w_time_utc").cast("long")) / 60.0,
                            F.lit(0.0),
                        ))
            .withColumn("weather_crosswind",
                        F.coalesce(F.abs(F.col("w_wspeed") * F.sin(wind_diff)), F.lit(0.0)))
            .withColumn("weather_headwind",
                        F.coalesce(F.col("w_wspeed") * F.cos(wind_diff), F.lit(0.0)))
            .withColumn("weather_icing_risk",
                        F.when(
                            (F.col("w_temp") <= 0.0) &
                            ((F.col("w_temp") - F.col("w_dew")) <= 2.0),
                            1.0,
                        ).otherwise(0.0))
            .withColumn("weather_ifr_conditions",
                        F.when(F.col("w_vis") < 4800.0, 1.0).otherwise(0.0))
        )

        # ── 3. Hurdle classifier score ────────────────────────────────────
        scored = hurdle_model.transform(enriched)
        output = (
            scored
            .withColumn("p_delayed", get_prob1("probability"))
            .withColumn("scored_at", F.current_timestamp().cast("string"))
            .select(
                "flight_key",
                "FlightDate",
                "Origin",
                "Dest",
                "Reporting_Airline",
                "scheduled_departure_utc",
                "p_delayed",
                "scored_at",
            )
        )

        n_out = output.count()
        logger.info("  Batch %d: scored %d flights.", batch_id, n_out)

        # ── 4a. Write to Kafka scores topic ───────────────────────────────
        if write_to_kafka:
            (
                output
                .select(
                    F.col("flight_key").cast("string").alias("key"),
                    F.to_json(F.struct("*")).alias("value"),
                )
                .write
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
                .option("topic", KAFKA_SCORES_TOPIC)
                .save()
            )
            logger.info("  Batch %d: published to Kafka topic '%s'.", batch_id, KAFKA_SCORES_TOPIC)

        # ── 4b. Parquet backup (always written) ───────────────────────────
        output.write.mode("append").parquet(str(output_dir))

    return process_batch


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_streaming_pipeline(
    spark: SparkSession,
    paths: dict = PATHS,
    task: str = "departure",
    source: str = "file",
    timeout_secs: int = 60,
) -> None:
    """
    Start the Structured Streaming pipeline.

    Parameters
    ----------
    spark        : active SparkSession (must be built with KAFKA_SPARK_PACKAGE
                   in spark.jars.packages when source='kafka')
    paths        : project paths dict
    task         : 'departure' or 'arrival'
    source       : 'kafka' — read from Kafka KAFKA_FLIGHT_TOPIC
                   'file'  — read JSON files from STREAMING_INPUT_DIR/flights/
    timeout_secs : seconds to run before stopping (use -1 to run indefinitely)
    """
    output_dir = Path(str(STREAMING_OUTPUT_DIR))
    checkpoint = Path(str(STREAMING_CHECKPOINT))
    input_dir  = Path(str(STREAMING_INPUT_DIR))
    for d in [output_dir, checkpoint, input_dir]:
        d.mkdir(parents=True, exist_ok=True)

    weather_parquet = paths["processed"] / f"noaa_hourly_{YEAR}.parquet"
    # Load bts_weather model — it only needs weather features which the
    # streaming pipeline derives from the as-of join. The full model
    # requires airport aggregates not available in real-time.
    clf_model_dir = paths["models"] / f"hurdle_{task}_bts_weather_classifier"
    if not clf_model_dir.exists():
        clf_model_dir = paths["models"] / f"hurdle_{task}_full_classifier"

    if not weather_parquet.exists():
        raise FileNotFoundError(
            f"Weather Parquet not found: {weather_parquet}. Run 'prep' first."
        )
    if not clf_model_dir.exists():
        raise FileNotFoundError(
            f"Hurdle classifier not found: {clf_model_dir}. Run 'hurdle' first."
        )

    logger.info("Loading static weather reference…")
    weather_df = _make_weather_reference(spark, weather_parquet)
    weather_df.cache()

    logger.info("Loading hurdle classifier from %s…", clf_model_dir)
    hurdle_model = PipelineModel.load(str(clf_model_dir))

    write_to_kafka = (source == "kafka")

    if source == "kafka":
        logger.info("Source: Kafka topic '%s' @ %s", KAFKA_FLIGHT_TOPIC, KAFKA_BOOTSTRAP_SERVERS)
        flight_stream = _make_kafka_flight_source(spark)
    else:
        logger.info("Source: file directory %s", input_dir)
        flight_stream = _make_file_flight_source(spark, input_dir)

    batch_fn = _make_batch_processor(
        weather_df, hurdle_model, output_dir, write_to_kafka, spark
    )

    query = (
        flight_stream.writeStream
        .foreachBatch(batch_fn)
        .option("checkpointLocation", str(checkpoint))
        .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info(
        "Streaming query started (id=%s, source=%s). "
        "%s",
        query.id,
        source,
        f"Publishing to Kafka topic '{KAFKA_SCORES_TOPIC}'." if write_to_kafka
        else f"Drop JSON files into {input_dir} to trigger scoring.",
    )

    if timeout_secs > 0:
        query.awaitTermination(timeout=timeout_secs)
        query.stop()
        logger.info("Streaming pipeline stopped after %ds.", timeout_secs)
    else:
        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            query.stop()
            logger.info("Streaming pipeline stopped by user.")

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
             Needs KAFKA_SPARK_PACKAGE on the classpath: set via
             spark.jars.packages with `python main.py`, or `--packages` with
             spark-submit (docker/submit.sh adds it for the stream step).
             Both topics are created on start if they do not exist yet.
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
    DELAY_AGG_FEATURE_COLS,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_FLIGHT_TOPIC,
    KAFKA_SCORES_TOPIC,
    PATHS,
    STREAMING_CHECKPOINT,
    STREAMING_INPUT_DIR,
    STREAMING_MODEL_CONDITION,
    STREAMING_OUTPUT_DIR,
    STREAMING_STATE_DIR,
    STREAMING_TRIGGER_SECONDS,
    YEAR,
)
from src.features.engineering import OPS_VALUE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema for a flight event message
# (must match what the producer serialises and what engineering.py produces)
# ---------------------------------------------------------------------------
# One topic carries two kinds of message (see src/streaming/producer.py):
#   event_type="flight" — a flight to score, sent at its prediction time T-2h
#   event_type="ops"    — an airport operational event (actual departure,
#                         actual arrival, flight starting to wait) used to
#                         rebuild the same-day airport delay features
FLIGHT_EVENT_SCHEMA = T.StructType([
    T.StructField("event_type",               T.StringType(),  True),
    T.StructField("flight_key",               T.LongType(),    True),
    T.StructField("FlightDate",               T.StringType(),  True),
    T.StructField("Reporting_Airline",        T.StringType(),  True),
    T.StructField("Origin",                   T.StringType(),  True),
    T.StructField("Dest",                     T.StringType(),  True),
    T.StructField("CRSElapsedTime",           T.DoubleType(),  True),
    T.StructField("scheduled_departure_utc",  T.StringType(),  True),  # ISO-8601, UTC
    T.StructField("origin_congestion_2h",     T.LongType(),    True),
    T.StructField("prior_leg_delay",          T.DoubleType(),  True),
    T.StructField("scheduled_month",          T.IntegerType(), True),
    T.StructField("scheduled_day_of_week",    T.IntegerType(), True),
    T.StructField("scheduled_departure_hour", T.DoubleType(),  True),
    # ops messages only
    T.StructField("ops_kind",                 T.StringType(),  True),
    T.StructField("event_time_sec",           T.LongType(),    True),
    *[T.StructField(c, T.DoubleType(), True) for c in OPS_VALUE_COLS],
])

# Schema of the scored output written to the Kafka scores topic / Parquet
SCORE_SCHEMA = T.StructType([
    T.StructField("flight_key",              T.LongType(),    True),
    T.StructField("FlightDate",              T.StringType(),  True),
    T.StructField("Origin",                  T.StringType(),  True),
    T.StructField("Dest",                    T.StringType(),  True),
    T.StructField("Reporting_Airline",       T.StringType(),  True),
    T.StructField("scheduled_departure_utc", T.StringType(),  True),
    T.StructField("p_delayed",               T.DoubleType(),  True),
    T.StructField("expected_delay",      T.DoubleType(),  True),
    T.StructField("model_condition",         T.StringType(),  True),
    T.StructField("scored_at",               T.StringType(),  True),
])


# ---------------------------------------------------------------------------
# Source factories
# ---------------------------------------------------------------------------

def _ensure_kafka_topics(topics: list, partitions: int = 3) -> None:
    """
    Create *topics* on the broker if they do not exist yet.

    The Spark Kafka source fails with UnknownTopicOrPartitionException when
    the stream starts before the producer has sent anything (the broker only
    auto-creates a topic on the first produce), so create them up front.
    Uses kafka-python (already required by the producer).
    """
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
    except ImportError:
        logger.warning("kafka-python not installed — cannot pre-create topics %s.", topics)
        return

    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    try:
        missing = [t for t in topics if t not in set(admin.list_topics())]
        for topic in missing:
            try:
                admin.create_topics([NewTopic(topic, num_partitions=partitions,
                                              replication_factor=1)])
                logger.info("Created Kafka topic '%s' (%d partitions).", topic, partitions)
            except TopicAlreadyExistsError:
                pass   # created concurrently by the producer — fine
    finally:
        admin.close()


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

def _load_hurdle_models(paths: dict, task: str, condition: str) -> tuple:
    """
    Load the hurdle classifier (+ regressor if present) for *condition*,
    falling back to bts_weather. Returns (condition, classifier, regressor|None).
    """
    for cond in dict.fromkeys([condition, "bts_weather"]):
        clf_dir = paths["models"] / f"hurdle_{task}_{cond}_classifier"
        if clf_dir.exists():
            reg_dir = paths["models"] / f"hurdle_{task}_{cond}_regressor"
            logger.info("Loading hurdle classifier from %s…", clf_dir)
            clf = PipelineModel.load(str(clf_dir))
            reg = None
            if reg_dir.exists():
                logger.info("Loading hurdle regressor from %s…", reg_dir)
                reg = PipelineModel.load(str(reg_dir))
            if cond != condition:
                logger.warning("Hurdle '%s' models not found — using '%s'.", condition, cond)
            return cond, clf, reg
    raise FileNotFoundError(
        f"No hurdle classifier for task '{task}' under {paths['models']}. Run 'hurdle' first."
    )


def _read_ops_state(spark: SparkSession, state_dir: Path):
    """All airport ops events received so far (deduplicated), or None."""
    if not state_dir.exists() or not any(state_dir.glob("*.parquet")):
        return None
    return (
        spark.read.parquet(str(state_dir))
        # a replay may resend the same events; count each once
        .dropDuplicates(["src_flight_key", "ops_kind"])
    )


def _make_batch_processor(
    weather_df: DataFrame,
    model_condition: str,
    clf_model: PipelineModel,
    reg_model,
    output_dir: Path,
    state_dir: Path,
    write_to_kafka: bool,
    spark: SparkSession,
    paths: dict,
):
    """
    Return a closure for ``writeStream.foreachBatch``.

    Each batch:
    1. Appends airport ops messages (event_type="ops") to the state table.
    2. For flight messages: as-of weather join [T-8h, T-2h] and aviation
       weather features (crosswind, headwind, icing, IFR).
    3. For the "full" model: same-day airport delay features from all ops
       events received so far with event time <= the flight's T-2h, plus
       storm counts and holiday flags — the same functions as batch
       feature engineering (src.features.engineering).
    4. Scores with the hurdle classifier (P(delay > 0)) and, if available,
       the regressor (expected delay = P × predicted length if delayed).
    5. Writes predictions to the Kafka scores topic (if enabled) and to a
       Parquet backup under output_dir.
    """
    import numpy as np
    from pyspark.sql.window import Window

    from src.features.engineering import (
        add_holiday_features,
        add_storm_features,
        same_day_airport_features,
    )

    AIRPORT_RUNWAYS = {
        "ATL":  90.0, "CLT": 180.0, "DEN": 170.0, "DFW": 175.0,
        "LAS": 260.0, "LAX": 250.0, "MCO": 180.0, "ORD": 270.0,
        "PHX": 260.0, "SEA": 160.0,
    }
    runway_map = F.create_map([F.lit(x) for kv in AIRPORT_RUNWAYS.items() for x in kv])
    rad = np.pi / 180.0
    get_prob1 = F.udf(lambda v: float(v[1]), T.DoubleType())
    use_full = model_condition == "full"
    ops_cols = ["src_flight_key", "ops_kind", "Origin", "FlightDate", "event_time_sec",
                *OPS_VALUE_COLS]

    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        batch_df = batch_df.persist()
        try:
            _process(batch_df, batch_id)
        finally:
            batch_df.unpersist()

    def _process(batch_df: DataFrame, batch_id: int) -> None:
        is_ops = F.col("event_type") == "ops"
        ops = batch_df.filter(is_ops)
        # Only well-formed flight messages are scored; anything else on the topic
        # (unknown event types, JSON that failed to parse) is ignored.
        flights = batch_df.filter(
            (F.col("event_type").isNull() | (F.col("event_type") == "flight"))
            & F.col("flight_key").isNotNull()
        )
        n_ops, n_flights = ops.count(), flights.count()
        if n_ops == 0 and n_flights == 0:
            logger.debug("Batch %d is empty — skipping.", batch_id)
            return
        logger.info("Batch %d: %d flights + %d airport ops events received.",
                    batch_id, n_flights, n_ops)

        # ── 1. Remember airport ops events ───────────────────────────────
        if n_ops:
            (ops.select(F.col("flight_key").alias("src_flight_key"), "ops_kind", "Origin",
                        F.to_date("FlightDate").alias("FlightDate"), "event_time_sec",
                        *OPS_VALUE_COLS)
             .select(*ops_cols)
             .write.mode("append").parquet(str(state_dir)))
        if n_flights == 0:
            return

        flights = (
            flights
            .withColumn("FlightDate", F.to_date("FlightDate"))
            .withColumn("t_minus_2_utc", F.col("event_time") - F.expr("INTERVAL 2 HOURS"))
        )

        # ── 2. As-of weather join [T-8h, T-2h] + aviation weather features ─
        t_minus_2 = F.col("t_minus_2_utc")
        join_cond = (
            (F.col("Origin")     == F.col("w_airport")) &
            (F.col("w_time_utc") <= t_minus_2) &
            (F.col("w_time_utc") >= t_minus_2 - F.expr("INTERVAL 6 HOURS"))
        )
        dedup_win = (
            Window.partitionBy("flight_key")
            .orderBy(F.col("w_time_utc").desc_nulls_last())
        )
        joined = (
            flights.join(weather_df, join_cond, "left")
            .withColumn("rn", F.row_number().over(dedup_win))
            .filter(F.col("rn") == 1)
            .drop("rn")
        )
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

        # ── 3. Same-day airport delays, storms, holidays (full model) ──────
        if use_full:
            queries = enriched.select(
                "flight_key", "Origin", "FlightDate",
                t_minus_2.cast("long").alias("query_time_sec"),
            )
            state = _read_ops_state(spark, state_dir)
            agg_cols = [c for c in DELAY_AGG_FEATURE_COLS]
            if state is not None:
                state = state.join(
                    queries.select("Origin", "FlightDate").distinct(),
                    on=["Origin", "FlightDate"], how="left_semi",
                )
                agg = same_day_airport_features(state, queries)
                enriched = enriched.join(agg, on="flight_key", how="left").fillna(0.0, subset=agg_cols)
            else:
                for c in agg_cols:
                    enriched = enriched.withColumn(c, F.lit(0.0))
            enriched = add_storm_features(spark, enriched, paths=paths)
            enriched = add_holiday_features(enriched, paths=paths)

        # ── 4. Hurdle score ────────────────────────────────────────────────
        enriched = enriched.cache()
        p_df = clf_model.transform(enriched).select(
            "flight_key", get_prob1("probability").alias("p_delayed"))
        output = enriched.select(
            "flight_key", F.col("FlightDate").cast("string").alias("FlightDate"),
            "Origin", "Dest", "Reporting_Airline", "scheduled_departure_utc",
        ).join(p_df, "flight_key")
        if reg_model is not None:
            len_df = reg_model.transform(enriched).select(
                "flight_key", F.expm1(F.greatest(F.col("prediction"), F.lit(0.0))).alias("_delay_if_delayed"))
            output = (
                output.join(len_df, "flight_key")
                .withColumn("expected_delay",
                            F.col("p_delayed") * F.col("_delay_if_delayed"))
                .drop("_delay_if_delayed")
            )
        else:
            output = output.withColumn("expected_delay", F.lit(None).cast("double"))
        output = (
            output
            .withColumn("model_condition", F.lit(model_condition))
            .withColumn("scored_at", F.current_timestamp().cast("string"))
        ).cache()

        n_out = output.count()
        logger.info("  Batch %d: scored %d flights (model: %s).", batch_id, n_out, model_condition)

        # ── 5a. Write to Kafka scores topic ───────────────────────────────
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

        # ── 5b. Parquet backup (always written) ───────────────────────────
        output.write.mode("append").parquet(str(output_dir))
        output.unpersist()
        enriched.unpersist()

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
    model_condition: str = STREAMING_MODEL_CONDITION,
) -> None:
    """
    Start the Structured Streaming pipeline.

    Parameters
    ----------
    spark           : active SparkSession (the Kafka connector must be on the
                      classpath when source='kafka')
    paths           : project paths dict
    task            : 'departure' or 'arrival'
    source          : 'kafka' — read from Kafka KAFKA_FLIGHT_TOPIC
                      'file'  — read JSON files from STREAMING_INPUT_DIR
    timeout_secs    : seconds to run before stopping (<= 0 runs until interrupted)
    model_condition : hurdle model to score with: 'full' (default) or 'bts_weather'
    """
    output_dir = Path(str(STREAMING_OUTPUT_DIR))
    checkpoint = Path(str(STREAMING_CHECKPOINT))
    input_dir  = Path(str(STREAMING_INPUT_DIR))
    state_dir  = Path(str(STREAMING_STATE_DIR))
    for d in [output_dir, checkpoint, input_dir, state_dir]:
        d.mkdir(parents=True, exist_ok=True)

    weather_parquet = paths["processed"] / f"noaa_hourly_{YEAR}.parquet"
    if not weather_parquet.exists():
        raise FileNotFoundError(
            f"Weather Parquet not found: {weather_parquet}. Run 'prep' first."
        )

    model_condition, clf_model, reg_model = _load_hurdle_models(paths, task, model_condition)

    logger.info("Loading static weather reference…")
    weather_df = _make_weather_reference(spark, weather_parquet)
    weather_df.cache()

    write_to_kafka = (source == "kafka")

    if source == "kafka":
        logger.info("Source: Kafka topic '%s' @ %s", KAFKA_FLIGHT_TOPIC, KAFKA_BOOTSTRAP_SERVERS)
        _ensure_kafka_topics([KAFKA_FLIGHT_TOPIC, KAFKA_SCORES_TOPIC])
        flight_stream = _make_kafka_flight_source(spark)
    else:
        logger.info("Source: file directory %s", input_dir)
        flight_stream = _make_file_flight_source(spark, input_dir)

    batch_fn = _make_batch_processor(
        weather_df, model_condition, clf_model, reg_model,
        output_dir, state_dir, write_to_kafka, spark, paths,
    )

    query = (
        flight_stream.writeStream
        .foreachBatch(batch_fn)
        .option("checkpointLocation", str(checkpoint))
        .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info(
        "Streaming query started (id=%s, source=%s, model=%s). %s",
        query.id,
        source,
        model_condition,
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

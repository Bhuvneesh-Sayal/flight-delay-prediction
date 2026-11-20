"""
src/streaming/producer.py
==========================
Replays historical BTS flight data into Kafka to simulate a live flight feed.

How it works
------------
1. Loads the processed BTS cohort Parquet (data/processed/bts_cohort_2025.parquet)
2. Sorts all flights by scheduled_departure_utc (chronological order)
3. Publishes each flight as a JSON message to the KAFKA_FLIGHT_TOPIC
4. Optionally throttles the replay to N events-per-second to simulate
   a realistic arrival rate without flooding the consumer

Message format
--------------
Each Kafka message:
  key   = flight_key (string)
  value = JSON object matching FLIGHT_EVENT_SCHEMA in pipeline.py

Fields required by the streaming pipeline that are not in the raw BTS Parquet
(origin_congestion_2h, prior_leg_delay) are computed on the fly using
the same logic as engineering.py so the streaming scores match batch scores.

Running
-------
  python main.py --steps produce                      # replay all flights
  python main.py --steps produce --produce-rate 100   # 100 events/sec
  python main.py --steps produce --produce-limit 1000 # only first 1000

Public API
----------
  run_producer(paths, bootstrap_servers, topic, rate_per_sec, limit) -> None
"""

import json
import time
from pathlib import Path

import pandas as pd

from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_FLIGHT_TOPIC,
    PATHS,
    TOP_10_AIRPORTS,
    YEAR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight congestion + rotation delay (pure pandas, no Spark needed)
# ---------------------------------------------------------------------------

def _add_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute origin_congestion_2h and prior_leg_delay purely in pandas.
    These match the definitions in engineering.py at a per-row level.

    Called once on the full replay DataFrame before the producer loop starts.
    """
    # Parse UTC timestamps
    df = df.copy()
    df["scheduled_departure_utc"] = pd.to_datetime(
        df["scheduled_departure_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["scheduled_departure_utc"])
    df = df.sort_values("scheduled_departure_utc").reset_index(drop=True)

    # --- origin_congestion_2h: count flights from same Origin in [T-2h, T) ---
    df["_dep_ts"] = df["scheduled_departure_utc"].astype("int64") // 10**9  # epoch seconds
    congestion = []
    for idx, row in df.iterrows():
        window_start = row["_dep_ts"] - 7200  # 2 hours in seconds
        window_end   = row["_dep_ts"]
        count = (
            (df["Origin"] == row["Origin"]) &
            (df["_dep_ts"] >= window_start) &
            (df["_dep_ts"] <  window_end)
        ).sum()
        congestion.append(int(count))
    df["origin_congestion_2h"] = congestion

    # --- prior_leg_delay: previous leg arrival delay for same Tail_Number ---
    # Guard: previous leg's scheduled arrival (dep + CRSElapsedTime) must be
    # known by T-2h of the current flight.
    tail_col = "Tail_Number" if "Tail_Number" in df.columns else None
    if tail_col:
        df = df.sort_values(["Tail_Number", "scheduled_departure_utc"])
        df["_prev_dep_ts"]   = df.groupby("Tail_Number")["_dep_ts"].shift(1)
        df["_prev_elapsed"]  = df.groupby("Tail_Number")["CRSElapsedTime"].shift(1)
        df["_prev_arr_delay"] = df.groupby("Tail_Number")["ArrDelay"].shift(1)

        df["_prev_sched_arr_ts"] = df["_prev_dep_ts"] + df["_prev_elapsed"] * 60.0
        t_minus_2_ts = df["_dep_ts"] - 7200

        mask = (
            df["_prev_sched_arr_ts"].notna() &
            (df["_prev_sched_arr_ts"] <= t_minus_2_ts) &
            (df["_prev_dep_ts"] < df["_dep_ts"])
        )
        df["prior_leg_delay"] = 0.0
        df.loc[mask, "prior_leg_delay"] = df.loc[mask, "_prev_arr_delay"].fillna(0.0)
        df = df.drop(columns=["_prev_dep_ts", "_prev_elapsed", "_prev_arr_delay",
                               "_prev_sched_arr_ts"])
    else:
        df["prior_leg_delay"] = 0.0

    df = df.drop(columns=["_dep_ts"])
    return df


# ---------------------------------------------------------------------------
# Main producer function
# ---------------------------------------------------------------------------

def run_producer(
    paths: dict = PATHS,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = KAFKA_FLIGHT_TOPIC,
    rate_per_sec: int = 50,
    limit: int = 0,
) -> None:
    """
    Replay historical BTS flight rows into Kafka in chronological order.

    Parameters
    ----------
    paths            : project paths dict
    bootstrap_servers: Kafka broker address (e.g. 'localhost:9092')
    topic            : Kafka topic to publish to
    rate_per_sec     : maximum events published per second (0 = no throttle)
    limit            : stop after publishing this many events (0 = all)
    """
    try:
        from kafka import KafkaProducer
        # NoBrokersAvailable name differs between kafka-python variants
        try:
            from kafka.errors import NoBrokersAvailable
        except ImportError:
            NoBrokersAvailable = Exception
    except ImportError:
        raise ImportError(
            "kafka-python is required for the producer. "
            "Install it with: pip install kafka-python"
        )

    # ── Load from feature set (has all required ML columns) ──────────────────
    feature_path = paths["features"] / "departure_bts_weather"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature set not found: {feature_path}. Run '--steps features' first."
        )

    logger.info("Loading feature set from %s…", feature_path)
    df = pd.read_parquet(feature_path)

    # Reconstruct scheduled_departure_utc from FlightDate + scheduled_departure_hour
    # (CRSDepTime is not stored in the feature set, but the hour is)
    if "scheduled_departure_utc" not in df.columns:
        dates = pd.to_datetime(df["FlightDate"].astype(str), errors="coerce")
        hour  = df["scheduled_departure_hour"].fillna(0).astype(int)
        df["scheduled_departure_utc"] = (
            dates + pd.to_timedelta(hour, unit="h")
        ).dt.strftime("%Y-%m-%dT%H:00:00")

    # CRSDepTime approximation from scheduled_departure_hour
    if "CRSDepTime" not in df.columns:
        df["CRSDepTime"] = (df["scheduled_departure_hour"].fillna(0) * 100).astype(int)

    # Ensure flight_key exists
    if "flight_key" not in df.columns:
        df["flight_key"] = range(len(df))

    # Sort chronologically
    df = df.sort_values("scheduled_departure_utc").reset_index(drop=True)

    if limit > 0:
        df = df.head(limit)
        logger.info("Limiting replay to first %d flights.", limit)

    total = len(df)
    logger.info("Connecting to Kafka at %s (topic: %s)…", bootstrap_servers, topic)

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            api_version=(2, 5, 0),   # skip broker version negotiation
            batch_size=16384,
            linger_ms=10,
            retries=3,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Kafka at '{bootstrap_servers}'. "
            "Make sure Kafka is running (docker compose up kafka) and the "
            "bootstrap server address is correct."
        ) from exc

    logger.info("Starting replay of %d flights at up to %d events/sec…",
                total, rate_per_sec if rate_per_sec > 0 else -1)

    # Columns to include in the Kafka message (matches FLIGHT_EVENT_SCHEMA)
    output_cols = [
        "flight_key", "FlightDate", "Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "CRSElapsedTime", "scheduled_departure_utc",
        "origin_congestion_2h", "prior_leg_delay",
        "scheduled_month", "scheduled_day_of_week", "scheduled_departure_hour",
    ]

    # flight_key column may not exist if cohort was saved before we added it
    if "flight_key" not in df.columns:
        df["flight_key"] = range(len(df))

    sent = 0
    batch_start = time.monotonic()
    batch_count = 0

    for _, row in df[output_cols].iterrows():
        msg = {
            "flight_key":               int(row["flight_key"]),
            "FlightDate":               str(row["FlightDate"]),
            "Reporting_Airline":        str(row["Reporting_Airline"]),
            "Origin":                   str(row["Origin"]),
            "Dest":                     str(row["Dest"]),
            "CRSDepTime":               int(row["CRSDepTime"]) if pd.notna(row["CRSDepTime"]) else 0,
            "CRSElapsedTime":           float(row["CRSElapsedTime"]) if pd.notna(row["CRSElapsedTime"]) else 0.0,
            "scheduled_departure_utc":  str(row["scheduled_departure_utc"]),
            "origin_congestion_2h":     int(row["origin_congestion_2h"]),
            "prior_leg_delay":          float(row["prior_leg_delay"]),
            "scheduled_month":          int(row["scheduled_month"]),
            "scheduled_day_of_week":    int(row["scheduled_day_of_week"]),
            "scheduled_departure_hour": float(row["scheduled_departure_hour"]),
        }

        producer.send(topic, key=msg["flight_key"], value=msg)
        sent += 1
        batch_count += 1

        # Throttle: sleep when we've sent `rate_per_sec` messages in < 1 second
        if rate_per_sec > 0 and batch_count >= rate_per_sec:
            elapsed = time.monotonic() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            batch_start = time.monotonic()
            batch_count = 0

        if sent % 1000 == 0:
            logger.info("  Published %d / %d flights…", sent, total)

    producer.flush()
    producer.close()
    logger.info("Producer finished. Published %d flights to '%s'.", sent, topic)

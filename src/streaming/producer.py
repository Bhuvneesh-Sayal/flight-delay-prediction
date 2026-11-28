"""
src/streaming/producer.py
==========================
Replays historical 2025 flights into Kafka to simulate a live feed.

How it works
------------
1. Loads the saved feature set (data/features/departure_full, or
   departure_bts_weather for older runs) and keeps one data split
   (default: test = Nov-Dec, the months the models never trained on).
2. Takes the first N flights in scheduled-departure order.
3. Loads the airport operational events of those flights
   (data/features/airport_ops_events: actual departures, actual arrivals,
   flights starting to wait) — built by engineering.build_airport_ops_events.
4. Publishes both kinds of message to KAFKA_FLIGHT_TOPIC in time order:
     event_type="flight" at the flight's prediction time T-2h
     event_type="ops"    at the moment the operational event happened
   so every event a flight's same-day airport features depend on reaches
   the stream before the flight itself.
5. Optionally throttles to N messages per second.

Message format
--------------
  key   = flight_key (string)
  value = JSON object matching FLIGHT_EVENT_SCHEMA in pipeline.py
          (flight messages: schedule + model inputs known in advance;
           ops messages: Origin, FlightDate, ops_kind, event_time_sec and
           the OPS_VALUE_COLS running-total contributions)

Running
-------
  python main.py --steps produce                        # all test flights
  python main.py --steps produce --produce-limit 1000   # first 1000 test flights
  python main.py --steps produce --produce-rate 100     # 100 messages/sec
  python main.py --steps produce --produce-split all    # all months

Public API
----------
  run_producer(paths, bootstrap_servers, topic, rate_per_sec, limit, split) -> None
"""

import json
import time

import pandas as pd

from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_FLIGHT_TOPIC,
    PATHS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Contribution columns of an ops event (same as engineering.OPS_VALUE_COLS;
# duplicated here so the producer does not need to import pyspark)
OPS_VALUE_COLS = [
    "dep_n", "del15_n", "dep_sum",
    "nas_n", "nas_min", "wx_n", "wx_min", "cr_n", "la_n", "la_min",
    "waiting",
]

# Flight fields sent to the stream (inputs known before departure)
_FLIGHT_INT = ["scheduled_month", "scheduled_day_of_week"]
_FLIGHT_FLOAT = ["CRSElapsedTime", "prior_leg_delay", "scheduled_departure_hour"]


def _to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Naive UTC timestamps (as Spark writes them) -> epoch seconds."""
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)


def _load_flights(paths: dict, split: str, limit: int, offset: int = 0) -> pd.DataFrame:
    for name in ("departure_full", "departure_bts_weather"):
        feature_path = paths["features"] / name
        if feature_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No feature set under {paths['features']}. Run '--steps prep features' first."
        )
    logger.info("Loading feature set from %s…", feature_path)
    # data_split is a Hive partition column, so this filter prunes whole
    # directories instead of reading every split into memory.
    filters = [("data_split", "==", split)] if split != "all" else None
    df = pd.read_parquet(feature_path, filters=filters)

    # Replay only one data split (partition column -> regular column in pandas)
    if split != "all":
        df = df[df["data_split"].astype(str) == split]
        if df.empty:
            raise ValueError(f"No flights with data_split='{split}' in {feature_path}.")
        logger.info("Replaying data_split='%s': %d flights (%s .. %s).", split, len(df),
                    df["FlightDate"].min(), df["FlightDate"].max())

    if "scheduled_departure_utc" in df.columns:
        df["sched_sec"] = _to_epoch_seconds(df["scheduled_departure_utc"])
    else:
        # Feature sets written before scheduled_departure_utc was saved:
        # fall back to the local scheduled hour (approximate).
        logger.warning("Feature set has no scheduled_departure_utc — rerun '--steps prep "
                       "features' for exact UTC times; using the local hour instead.")
        local = pd.to_datetime(df["FlightDate"].astype(str)) + pd.to_timedelta(
            df["scheduled_departure_hour"].fillna(0).astype(int), unit="h")
        df["sched_sec"] = (local - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    df = df.sort_values(["sched_sec", "flight_key"]).reset_index(drop=True)
    if offset > 0:
        df = df.iloc[offset:]
    if limit > 0:
        df = df.head(limit)
        logger.info("Limiting replay to %d flights (starting at #%d).", limit, offset)
    return df.reset_index(drop=True)


def _flight_messages(df: pd.DataFrame) -> list:
    msgs = []
    for row in df.itertuples(index=False):
        sched_iso = pd.Timestamp(row.sched_sec, unit="s").strftime("%Y-%m-%dT%H:%M:%S")
        msg = {
            "event_type": "flight",
            "flight_key": int(row.flight_key),
            "FlightDate": str(row.FlightDate)[:10],
            "Reporting_Airline": str(row.Reporting_Airline),
            "Origin": str(row.Origin),
            "Dest": str(row.Dest),
            "scheduled_departure_utc": sched_iso,
            "origin_congestion_2h": int(row.origin_congestion_2h),
        }
        for c in _FLIGHT_INT:
            msg[c] = int(getattr(row, c))
        for c in _FLIGHT_FLOAT:
            v = getattr(row, c)
            msg[c] = float(v) if pd.notna(v) else None
        # sent at the prediction time T-2h
        msgs.append((int(row.sched_sec) - 7200, 1, msg))
    return msgs


def _ops_messages(paths: dict, flight_keys: set) -> list:
    ops_path = paths["features"] / "airport_ops_events"
    if not ops_path.exists():
        logger.warning("No %s — sending flights only; same-day airport features "
                       "will be 0 in the stream. Rerun '--steps prep features'.", ops_path)
        return []
    ops = pd.read_parquet(ops_path)
    ops = ops[ops["src_flight_key"].isin(flight_keys)]
    msgs = []
    for row in ops.itertuples(index=False):
        msg = {
            "event_type": "ops",
            "flight_key": int(row.src_flight_key),
            "ops_kind": str(row.ops_kind),
            "Origin": str(row.Origin),
            "FlightDate": str(row.FlightDate)[:10],
            "event_time_sec": int(row.event_time_sec),
        }
        for c in OPS_VALUE_COLS:
            msg[c] = float(getattr(row, c))
        # ops before flights at the same second: "at or before T-2h" counts
        msgs.append((int(row.event_time_sec), 0, msg))
    return msgs


def build_replay_messages(
    paths: dict = PATHS,
    limit: int = 50,
    offset: int = 0,
    split: str = "test",
) -> list:
    """
    Return *limit* flights (from flight #*offset* in scheduled order) plus their
    airport ops events as a list of JSON-serialisable message dicts, ordered by
    event time (ops before flights at the same second).  Used by the Streamlit
    "batch streaming" tab; needs no Kafka or Spark.
    """
    flights = _load_flights(paths, split, limit, offset)
    messages = _flight_messages(flights) + _ops_messages(paths, set(flights["flight_key"]))
    messages.sort(key=lambda m: (m[0], m[1]))
    return [m[2] for m in messages]


def run_producer(
    paths: dict = PATHS,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = KAFKA_FLIGHT_TOPIC,
    rate_per_sec: int = 50,
    limit: int = 0,
    split: str = "test",
) -> None:
    """
    Replay flights and their airport operational events into Kafka in time order.

    Parameters
    ----------
    paths            : project paths dict
    bootstrap_servers: Kafka broker address (e.g. 'kafka:9092')
    topic            : Kafka topic to publish to
    rate_per_sec     : maximum messages published per second (0 = no throttle)
    limit            : number of flights to replay (0 = all in the split)
    split            : 'test' (default, Nov-Dec), 'validation', 'train' or 'all'
    """
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise ImportError(
            "kafka-python is required for the producer. "
            "Install it with: pip install kafka-python"
        )

    flights = _load_flights(paths, split, limit)
    messages = _flight_messages(flights) + _ops_messages(paths, set(flights["flight_key"]))
    messages.sort(key=lambda m: (m[0], m[1]))
    n_flights = len(flights)
    n_ops = len(messages) - n_flights
    logger.info("Prepared %d flight messages + %d airport ops messages.", n_flights, n_ops)

    logger.info("Connecting to Kafka at %s (topic: %s)…", bootstrap_servers, topic)
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            batch_size=16384,
            linger_ms=10,
            retries=3,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Kafka at '{bootstrap_servers}'. "
            "Make sure Kafka is running and the bootstrap server address is correct."
        ) from exc

    logger.info("Starting replay of %d messages at up to %s messages/sec…",
                len(messages), rate_per_sec if rate_per_sec > 0 else "unlimited")

    sent = 0
    batch_start = time.monotonic()
    batch_count = 0
    for _, _, msg in messages:
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

        if sent % 5000 == 0:
            logger.info("  Published %d / %d messages…", sent, len(messages))

    producer.flush()
    producer.close()
    logger.info("Producer finished. Published %d flights + %d ops events to '%s'.",
                n_flights, n_ops, topic)

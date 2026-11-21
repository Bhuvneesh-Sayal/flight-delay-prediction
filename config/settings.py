"""
config/settings.py
==================
Central configuration for the Flight Delay Prediction project.
All constants, paths, and domain mappings live here so that every
module imports from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — always the repo root (parent of this config/ directory)
# ---------------------------------------------------------------------------
PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data year
# ---------------------------------------------------------------------------
YEAR: int = 2025

# ---------------------------------------------------------------------------
# Paths  (all relative to PROJECT_DIR so they work on Drive and locally)
# ---------------------------------------------------------------------------
PATHS: dict = {
    "bts_raw":    PROJECT_DIR / "data" / "raw"  / "bts",
    "noaa_raw":   PROJECT_DIR / "data" / "raw"  / "noaa",
    "storms_raw": PROJECT_DIR / "data" / "raw"  / "storms",
    "interim":    PROJECT_DIR / "data" / "interim",
    "processed":  PROJECT_DIR / "data" / "processed",
    "features":   PROJECT_DIR / "data" / "features",
    "models":     PROJECT_DIR / "models",
    "results":    PROJECT_DIR / "results",
    "benchmarks": PROJECT_DIR / "benchmarks_output",
    "logs":       PROJECT_DIR / "logs",
}

# Ensure all directories exist at import time
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Remote data URLs
# ---------------------------------------------------------------------------
BTS_BASE_URL: str  = "https://transtats.bts.gov/PREZIP"
NOAA_BASE_URL: str = (
    f"https://www.ncei.noaa.gov/oa/global-historical-climatology-network"
    f"/hourly/access/by-year/{YEAR}/psv"
)

# NOAA Storm Events — severe weather (thunderstorms, tornadoes, hail, floods)
# File naming convention changes each release; discover dynamically in ingestion.
STORM_EVENTS_DIR_URL: str = (
    "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
)

# US public holidays API (no auth needed)
HOLIDAYS_API_URL: str = f"https://date.nager.at/api/v3/PublicHolidays/{YEAR}/US"

# ---------------------------------------------------------------------------
# Closed-cohort airport selection
# ---------------------------------------------------------------------------
TOP_10_AIRPORTS: list = ["ATL", "CLT", "DEN", "DFW", "LAS", "LAX", "MCO", "ORD", "PHX", "SEA"]

# NOAA station IDs for each hub airport
AIRPORT_STATIONS: dict = {
    "ATL": "USW00013874",
    "CLT": "USW00013881",
    "DEN": "USW00003017",
    "DFW": "USW00003927",
    "LAS": "USW00023169",
    "LAX": "USW00023174",
    "MCO": "USW00012815",
    "ORD": "USW00094846",
    "PHX": "USW00023183",
    "SEA": "USW00024233",
}

# Primary runway headings (degrees magnetic) used for headwind / crosswind math
AIRPORT_RUNWAYS: dict = {
    "ATL":  90.0,
    "CLT": 180.0,
    "DEN": 170.0,
    "DFW": 175.0,
    "LAS": 260.0,
    "LAX": 250.0,
    "MCO": 180.0,
    "ORD": 270.0,
    "PHX": 260.0,
    "SEA": 160.0,
}

# Airport coordinates (lat, lon) for storm-event proximity matching
AIRPORT_COORDS: dict = {
    "ATL": (33.64, -84.43),
    "CLT": (35.21, -80.94),
    "DEN": (39.86, -104.67),
    "DFW": (32.90, -97.04),
    "LAS": (36.08, -115.15),
    "LAX": (33.94, -118.41),
    "MCO": (28.43, -81.31),
    "ORD": (41.97, -87.91),
    "PHX": (33.44, -112.01),
    "SEA": (47.45, -122.31),
}

# Storm proximity radius in degrees (~50 km ≈ 0.45°)
STORM_RADIUS_DEG: float = 0.45

# ---------------------------------------------------------------------------
# BTS columns to keep from the raw CSV files
# ---------------------------------------------------------------------------
BTS_KEEP_COLS: list = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",          # may be "TailNum" in older downloads — handled in preprocessing
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    # Delay-cause breakdown (populated only for delayed flights; null otherwise)
    "CarrierDelay",         # minutes attributed to carrier (maintenance, crew, baggage)
    "WeatherDelay",         # minutes attributed to significant weather
    "NASDelay",             # minutes attributed to NAS (ATC, airport ops, ground stops)
    "SecurityDelay",        # minutes attributed to security events
    "LateAircraftDelay",    # minutes attributed to cascading late-arriving aircraft
]

# NOAA columns to keep from the raw PSV files
NOAA_KEEP_COLS: list = [
    "DATE",
    "temperature",
    "dew_point_temperature",
    "wind_direction",
    "wind_speed",
    "visibility",
    "sea_level_pressure",
]

# ---------------------------------------------------------------------------
# Feature column sets
# ---------------------------------------------------------------------------
BTS_FEATURE_COLS: list = [
    "Origin",
    "Dest",
    "Reporting_Airline",
    "scheduled_month",
    "scheduled_day_of_week",
    "scheduled_departure_hour",
    "CRSElapsedTime",
    "origin_congestion_2h",
    "prior_leg_delay",
]

WEATHER_FEATURE_COLS: list = BTS_FEATURE_COLS + [
    "w_temp",
    "w_dew",
    "w_wspeed",
    "w_slp",
    "weather_crosswind",
    "weather_headwind",
    "weather_icing_risk",
    "weather_ifr_conditions",
    "weather_available",
    "weather_age_minutes",
]

# ---------------------------------------------------------------------------
# Enhanced feature columns — BTS delay-cause aggregates + storms + holidays
# ---------------------------------------------------------------------------
# Same-day airport delay picture at the origin, as known at T-2h (no leakage).
# Departure delays count once a flight has actually departed by T-2h;
# delay-cause minutes count once it has actually landed by T-2h.
DELAY_AGG_FEATURE_COLS: list = [
    "airport_nas_delays_today",         # flights landed by T-2h with NAS delay
    "airport_nas_delay_min_today",      # their NASDelay minutes
    "airport_wx_delays_today",          # flights landed by T-2h with weather delay
    "airport_wx_delay_min_today",       # their WeatherDelay minutes
    "airport_carrier_delays_today",     # flights landed by T-2h with carrier delay
    "airport_late_acft_delays_today",   # flights landed by T-2h with late-aircraft delay
    "airport_late_acft_min_today",      # their LateAircraftDelay minutes
    "airport_total_delayed_today",      # flights departed by T-2h with DepDelay > 15
    "airport_avg_delay_today",          # mean DepDelay of flights departed by T-2h
    "airport_delay_rate_today",         # share of those departed > 15 min late
    "airport_waiting_late_now",         # flights past scheduled departure, not yet left, at T-2h
]

# NOAA Storm Events + US holidays
CONTEXTUAL_FEATURE_COLS: list = [
    "storm_events_24h",                 # severe weather events near airport in past 24h
    "storm_events_6h",                  # severe weather events near airport in past 6h
    "is_holiday",                       # 1 if departure date is a US federal holiday
    "is_holiday_adjacent",              # 1 if within ±1 day of a holiday
]

# Full enhanced set: weather + delay aggregates + storms + holidays
FULL_FEATURE_COLS: list = (
    WEATHER_FEATURE_COLS + DELAY_AGG_FEATURE_COLS + CONTEXTUAL_FEATURE_COLS
)

# ---------------------------------------------------------------------------
# Chronological data-split boundaries (months are inclusive)
# ---------------------------------------------------------------------------
SPLIT_TRAIN_MONTHS:      tuple = (1, 8)   # January – August
SPLIT_VALIDATION_MONTHS: tuple = (9, 10)  # September – October
SPLIT_TEST_MONTHS:       tuple = (11, 12) # November – December

# ---------------------------------------------------------------------------
# Spark configuration
# ---------------------------------------------------------------------------
SPARK_CONFIG: dict = {
    "spark.sql.shuffle.partitions":                 "32",
    "spark.sql.adaptive.enabled":                   "true",
    "spark.sql.adaptive.coalescePartitions.enabled":"true",
    "spark.driver.memory":                          "8g",
    "spark.driver.maxResultSize":                   "4g",
}

# ---------------------------------------------------------------------------
# Model hyper-parameters (ablation defaults)
# ---------------------------------------------------------------------------
LR_PARAMS: dict   = {"maxIter": 25, "regParam": 0.1}
RF_PARAMS: dict   = {"numTrees": 40, "maxDepth": 8,  "maxBins": 64, "seed": 42}
GBT_PARAMS: dict  = {"maxIter": 25, "maxDepth": 6,  "maxBins": 64, "seed": 42}

# Hyperparameter tuning grids (used by ablation cross-validation)
LR_PARAM_GRID: dict = {
    "regParam": [0.01, 0.1, 1.0],
    "maxIter":  [25, 50],
}
RF_PARAM_GRID: dict = {
    "numTrees": [20, 40],
    "maxDepth": [6, 8],
}
GBT_PARAM_GRID: dict = {
    "maxIter":  [15, 25],
    "maxDepth": [4, 6],
}

# Cross-validation folds
CV_FOLDS: int = 3

# Minimum weather coverage rate (fraction) below which weather features are flagged
MIN_WEATHER_COVERAGE: float = 0.5

# Delay boundary used consistently across ablation and hurdle models
DELAY_POSITIVE_THRESHOLD: float = 0.0   # regressor trained on delay > this
DELAY_BINARY_THRESHOLD:   float = 0.0   # classifier trained on delay > this (aligns with regressor)

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
BENCHMARK_CORES:    list = [1, 2, 4]
STRONG_SCALE_ROWS:  int  = 300_000
WEAK_SCALE_ROWS_PER_CORE: int = 100_000

# Spark standalone cluster URL (used when running inside Docker)
# Override via SPARK_MASTER_URL environment variable in production.
# Validated against a strict allowlist pattern to prevent redirection
# to a rogue master via environment manipulation.
import os as _os
import re as _re

_raw_master_url: str = _os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
_MASTER_URL_PATTERN = _re.compile(
    r"^spark://[a-zA-Z0-9._-]+(:\d{1,5})?$"
)
if not _MASTER_URL_PATTERN.match(_raw_master_url):
    raise ValueError(
        f"SPARK_MASTER_URL '{_raw_master_url}' does not match the expected "
        "pattern 'spark://<host>[:<port>]'. "
        "Set a valid value or unset the environment variable."
    )
CLUSTER_MASTER_URL: str = _raw_master_url

# ---------------------------------------------------------------------------
# Structured Streaming settings
# ---------------------------------------------------------------------------
STREAMING_WATERMARK_MINUTES: int = 15      # late-event tolerance
STREAMING_TRIGGER_SECONDS:   int = 5       # micro-batch interval (5 s for low-latency demo)
STREAMING_INPUT_DIR:  Path = PROJECT_DIR / "data" / "streaming" / "input"  / "flights"
STREAMING_OUTPUT_DIR: Path = PROJECT_DIR / "data" / "streaming" / "output" / "scores"
STREAMING_CHECKPOINT: Path = PROJECT_DIR / "data" / "streaming" / "checkpoints"

# ---------------------------------------------------------------------------
# Kafka settings  (KRaft mode — no Zookeeper)
# ---------------------------------------------------------------------------
# Bootstrap server used by both the Python producer and the Spark consumer.
# Outside Docker (local dev): localhost:9092
# Inside Docker Compose (Spark driver talks to kafka service): kafka:9092
KAFKA_BOOTSTRAP_SERVERS: str = _os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Topic the producer publishes flight events to and Spark reads from
KAFKA_FLIGHT_TOPIC: str  = "flight-events"

# Topic Spark writes scored predictions to
KAFKA_SCORES_TOPIC: str  = "delay-predictions"

# Spark SQL Kafka connector Maven coordinate.
# Must match the PySpark major version (4.0.x) and Scala version (2.13 for PySpark 4.x).
KAFKA_SPARK_PACKAGE: str = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0"

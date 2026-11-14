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
    "bts_raw":   PROJECT_DIR / "data" / "raw"  / "bts",
    "noaa_raw":  PROJECT_DIR / "data" / "raw"  / "noaa",
    "interim":   PROJECT_DIR / "data" / "interim",
    "processed": PROJECT_DIR / "data" / "processed",
    "features":  PROJECT_DIR / "data" / "features",
    "models":    PROJECT_DIR / "models",
    "results":   PROJECT_DIR / "results",
    "benchmarks":PROJECT_DIR / "benchmarks_output",
    "logs":      PROJECT_DIR / "logs",
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
import os as _os
CLUSTER_MASTER_URL: str = _os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")

# ---------------------------------------------------------------------------
# Structured Streaming settings
# ---------------------------------------------------------------------------
STREAMING_WATERMARK_MINUTES: int = 15      # late-event tolerance
STREAMING_TRIGGER_SECONDS:   int = 30      # micro-batch interval
STREAMING_INPUT_DIR:  Path = PROJECT_DIR / "data" / "streaming" / "input"  / "flights"
STREAMING_OUTPUT_DIR: Path = PROJECT_DIR / "data" / "streaming" / "output" / "scores"
STREAMING_CHECKPOINT: Path = PROJECT_DIR / "data" / "streaming" / "checkpoints"

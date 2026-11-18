# Flight Delay Prediction

Scalable flight delay prediction using **PySpark**, built as part of the
**DA231o — Data Scaling** MTech course project.

The project ingests a full year of US domestic flight data from the Bureau of
Transportation Statistics (BTS) and paired hourly weather observations from
NOAA, then trains and evaluates regression and classification models at scale.

---

## Project Structure

```
flight-delay-prediction/
│
├── main.py                      # Pipeline entry point (structured logging,
│                                # error handling, run all or specific steps)
│
├── config/
│   └── settings.py              # Constants, paths, feature lists, hyper-params,
│                                # tuning grids, thresholds, streaming settings
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Download helper (cache-first, retries, SHA-256 audit)
│   │   └── preprocessing.py     # BTS cleaning, NOAA standardisation, quality checks
│   ├── features/
│   │   └── engineering.py       # UTC conversion, congestion, rotation delay,
│   │                            # as-of weather join, aviation features, labels
│   ├── models/
│   │   ├── pipeline.py          # Reusable Spark ML pipeline builder
│   │   ├── ablation.py          # Baseline + controlled ablation with CV tuning
│   │   └── hurdle.py            # Two-stage hurdle model (classifier + regressor)
│   ├── analysis/
│   │   ├── feature_importance.py # RF/GBT importances → CSV + bar charts
│   │   └── residuals.py          # Error distributions, QQ plots, hurdle calibration
│   ├── benchmarks/
│   │   └── scaling.py           # Strong (Amdahl) + weak (Gustafson) benchmarks
│   ├── streaming/
│   │   └── pipeline.py          # Spark Structured Streaming, 15-min watermark
│   └── utils/
│       └── logger.py            # get_logger(): console + logs/pipeline.log
│
├── app/
│   └── streamlit_app.py         # Interactive delay inference UI
│
├── tests/                       # Unit tests (pytest)
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_pipeline_builder.py
│   └── test_settings.py
│
├── data/
│   ├── raw/bts/                 # Downloaded BTS ZIPs and CSVs
│   ├── raw/noaa/                # Downloaded NOAA PSV files
│   ├── processed/               # Cleaned Parquet files
│   ├── features/                # Partitioned feature sets
│   └── streaming/               # Streaming input / output / checkpoints
│
├── models/                      # Saved model artefacts
├── results/                     # Experiment result CSVs + all plots
├── benchmarks_output/           # Benchmark CSVs + scaling curve PNGs
├── logs/                        # Download manifests, pipeline.log
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quickstart

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode (makes src/ importable)
pip install -e .

# 4. Run the full pipeline
python main.py

# 4a. Or run specific steps only
python main.py --steps ingest prep features
python main.py --steps ablation hurdle
python main.py --steps analysis
python main.py --steps benchmarks
python main.py --steps stream

# 5. Launch the Streamlit demo (requires hurdle step to have run first)
streamlit run app/streamlit_app.py
```

---

## Data Sources

| Source | Description | Volume |
|--------|-------------|--------|
| [BTS On-Time Performance](https://transtats.bts.gov/PREZIP) | Monthly flight-level on-time statistics, 2025 | ~12 ZIPs |
| [NOAA GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) | Hourly surface weather for 10 hub airports, 2025 | 10 PSV files |

**Closed-cohort filter**: only flights where *both* Origin and Destination
are among the 10 busiest US hub airports are retained.

Airports: `ATL CLT DEN DFW LAS LAX MCO ORD PHX SEA`

---

## Pipeline Steps

### 1. Data Ingestion (`src/data/ingestion.py`)
- Cache-first download with atomic `.part` temp files
- **Retry logic**: 5 retries with exponential backoff (5 s, 10 s, 20 s, 40 s,
  80 s), then a `RuntimeError`; partial files are cleaned up
- SHA-256 checksum logged for every file
- BTS: ZIPs downloaded, largest CSV member extracted
- NOAA: PSV files downloaded per station

### 2. Preprocessing (`src/data/preprocessing.py`)
- **Flights**: closed-cohort filter, type casts, calendar features,
  drop cancelled/diverted rows
- **Weather**: UTC timestamp parse, year filter, deduplication, ISO string
  conversion (avoids PyArrow INT64 overflow)
- **Data quality checks** (`run_quality_checks`): logs total rows, null rates
  (`DepDelay`, `ArrDelay`, `CRSElapsedTime`, `Tail_Number`), delay distribution
  (mean, stddev, min, max, p50, p95) and train/validation/test row counts
  when a split column is present
- Output: Snappy-compressed Parquet for both datasets

### 3. Feature Engineering (`src/features/engineering.py`)
| Feature | Description |
|---------|-------------|
| `origin_congestion_2h` | Rolling 2-hour departure count at origin airport |
| `prior_leg_delay` | Previous arrival delay for the same aircraft tail number (only legs that departed before the current scheduled departure) |
| `weather_crosswind` | Wind component perpendicular to runway heading |
| `weather_headwind` | Wind component along runway heading |
| `weather_icing_risk` | 1 if temp ≤ 0°C and dew-point spread ≤ 2°C |
| `weather_ifr_conditions` | 1 if visibility < 4,800 m |

- **Weather join coverage**: after the as-of join, logs total flights, the
  share with weather, and a per-origin-airport breakdown. It warns when
  coverage falls below `MIN_WEATHER_COVERAGE`.
- **Chronological split**: Jan–Aug train / Sep–Oct validation / Nov–Dec test

### 4. Ablation Experiments (`src/models/ablation.py`)
3 algorithms × 3 feature conditions × 2 targets = **18 models**, plus a
**mean baseline** (train-set mean) for each target and condition.

- **Validation evaluation**: each row reports `Val_MAE_min`, `Val_RMSE_min`
  and `Val_R2` next to the test metrics
- **Cross-validation tuning**: `CrossValidator` (`CV_FOLDS` folds, scored on
  MAE) over `LR_PARAM_GRID`, `RF_PARAM_GRID` and `GBT_PARAM_GRID`; best
  params and CV MAE are logged
- **Model persistence**: best models saved to
  `models/{task}_{condition}_{algorithm}`
- Pipelines impute only numeric columns that have nulls in the training data

| Algorithm | BTS-only MAE | BTS+Weather MAE |
|-----------|-------------|-----------------|
| LinearRegression | 31–36 min | 29–33 min |
| RandomForest | 26–30 min | 25–29 min |
| GBT | 25–30 min | 25–29 min |

*Figures are from the earlier untuned run; re-run `--steps ablation` for
current numbers in `results/`.*

### 5. Two-Stage Hurdle Model (`src/models/hurdle.py`)
- **Stage 1**: Logistic Regression classifies P(delay > 0), with the label
  set by `DELAY_BINARY_THRESHOLD`
- **Stage 2**: GBT Regressor on log1p(delay), trained on flights with
  delay > `DELAY_POSITIVE_THRESHOLD`
- **Aligned boundary**: both stages use the same delay > 0 definition, so the
  classifier's population matches what the regressor was trained on
- **Combined**: E[delay] = P(delay > 0) × E[delay | delay > 0]
- **Model persistence**: saved as
  `models/hurdle_{task}_{condition}_classifier` and
  `models/hurdle_{task}_{condition}_regressor`
- An earlier version, which used a 15-minute classifier boundary, improved
  departure MAE from 25.15 → 21.35 min (ROC-AUC ≈ 0.67). Re-run
  `--steps hurdle` to refresh these figures.

### 6. Scaling Benchmarks (`src/benchmarks/scaling.py`)
| Type | Cores | Time (s) | Speedup |
|------|-------|----------|---------|
| Strong | 1 | 2.84 | 1.00× |
| Strong | 2 | 1.77 | 1.61× |
| Strong | 4 | 1.27 | 2.24× |
| Weak | 1 | 1.09 | — |
| Weak | 2 | 1.08 | — |
| Weak | 4 | 1.39 | — |

### 7. Feature Importance Analysis (`src/analysis/feature_importance.py`)
- Loads every saved RF and GBT model after the ablation step
- Extracts `featureImportances` SparseVector from each pipeline's last stage
- Maps vector indices back to named features via VectorAssembler metadata
- Saves a combined `results/feature_importances.csv` and one bar chart PNG
  per model (top 20 features, sorted by importance)

### 8. Residual Analysis (`src/analysis/residuals.py`)
- Runs on a 10% sample of the test partition for speed
- Per model: **error distribution histogram**, **predicted vs actual scatter**,
  **QQ plot** of standardised residuals against Normal quantiles
- **Hurdle calibration curve**: bins predicted P(delay > 0) into 10 equal
  buckets and plots mean predicted probability vs actual delay fraction — a
  reliability diagram for the classifier stage

### 9. Structured Streaming (`src/streaming/pipeline.py`)
- Reads flight events as JSON from `data/streaming/input/flights/`
- Applies a **15-minute watermark** on `scheduled_departure_utc` to tolerate
  late-arriving events
- Joins each micro-batch with a static weather reference DataFrame (as-of join
  in the [T-8h, T-2h] window)
- Re-derives aviation features (crosswind, headwind, icing, IFR) per batch
- Scores with the saved hurdle classifier and writes predictions to
  `data/streaming/output/scores/`
- Checkpoint at `data/streaming/checkpoints/`; trigger interval 30 s
- **To swap to Kafka in production**: replace `_make_flight_source` with a
  `readStream.format("kafka")` call — no other logic changes

---

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Requires `python main.py --steps hurdle` to have been run first (saves the
hurdle models to `models/`).

The UI lets you configure origin, destination, airline, date/time, and weather
conditions, then returns:
- **P(Departure Delayed)** — from the hurdle classifier
- **Expected Departure Delay (min)** — combined two-stage prediction
- Colour-coded risk assessment and a model details breakdown

---

## Docker Cluster (3-node Spark)

Runs a real Spark standalone cluster (1 master + 3 workers) inside Docker for
inter-node network shuffle benchmarking.  Requires Docker Desktop ≥ 4.x.

### Setup

```bash
# 1. Build the shared image (Python 3.11 + Spark 3.5.1 + all deps)
docker compose build

# 2. Start master + 3 workers in the background
docker compose up -d

# 3. Verify all 3 workers registered:
#    open http://localhost:8080 — should show 3 workers, 6 cores, 12 GB RAM
```

### Submit a job

```bash
# Using the convenience script (from repo root)
./docker/submit.sh --steps cluster_bench

# Or use docker compose run directly
docker compose run --rm --profile driver driver \
    python main.py --steps cluster_bench
```

### Web UIs

| URL | Service |
|-----|---------|
| http://localhost:8080 | Spark Master (cluster overview) |
| http://localhost:8081 | Worker 1 |
| http://localhost:8082 | Worker 2 |
| http://localhost:8083 | Worker 3 |
| http://localhost:4040 | Spark App UI (only while a job runs) |

### Teardown

```bash
docker compose down
```

### What the cluster benchmark measures (`src/benchmarks/cluster_scaling.py`)

Three workloads are run at three data sizes (100K / 300K / 465K rows):

| Workload | What it stresses |
|----------|-----------------|
| `ShuffleAgg` | Full data redistribution across all 3 workers (groupBy + 5 aggs) |
| `GlobalSort` | Network shuffle sort stage (orderBy on 3 columns) |
| `HashJoin`   | Inter-node join performance (hash join, no broadcast) |

Results saved to `benchmarks_output/cluster_scaling_results.csv` and
`benchmarks_output/cluster_scaling_chart.png`.

### Key differences vs. local benchmarks

| | Local (`--steps benchmarks`) | Cluster (`--steps cluster_bench`) |
|--|---|---|
| Parallelism | Simulated (`local[P]`) | Real inter-node network shuffles |
| Workers | 1 JVM, P threads | 3 Docker containers |
| Shuffle | In-process | TCP over `spark-net` bridge |
| Use case | Amdahl/Gustafson analysis | Actual cluster overhead measurement |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

| File | Covers |
|------|--------|
| `test_ingestion.py` | Cache hits, download, SHA-256 manifest, retry with backoff |
| `test_preprocessing.py` | Weather standardisation: year filter, deduplication per airport |
| `test_pipeline_builder.py` | Stage composition of `build_pipeline` (needs Java for a local Spark session) |
| `test_settings.py` | Settings sanity: split months, grids, thresholds, CV folds |

---

## Configuration

All tunable parameters are in `config/settings.py`:

- `YEAR` — data year (default 2025)
- `TOP_10_AIRPORTS` — cohort definition
- `SPARK_CONFIG` — Spark session settings
- `LR_PARAMS / RF_PARAMS / GBT_PARAMS` — model hyper-parameters
- `LR_PARAM_GRID / RF_PARAM_GRID / GBT_PARAM_GRID` — cross-validation search grids (estimator param name → list of values)
- `CV_FOLDS` — number of cross-validation folds
- `DELAY_POSITIVE_THRESHOLD` — minimum delay for a flight to enter the hurdle regressor
- `DELAY_BINARY_THRESHOLD` — delay above which the hurdle classifier label is 1
- `MIN_WEATHER_COVERAGE` — weather join coverage below this logs a warning
- `SPLIT_*_MONTHS` — chronological split boundaries
- `BENCHMARK_CORES / STRONG_SCALE_ROWS / WEAK_SCALE_ROWS_PER_CORE` — benchmark settings

---

## Logging

All modules log through `src.utils.logger.get_logger`. Messages go to the
console (INFO and above) and are appended to `logs/pipeline.log`, so repeated
runs accumulate in one file. The log records download retries and checksums,
data quality statistics, weather join coverage, best CV parameters and
errors. `main.py` logs unhandled exceptions with a traceback and exits with
status 1 (130 on Ctrl-C).

---

## Notes on Model Performance

R² values across all models are low (≈ 0.05).  This is expected for flight
delay regression — delay is driven by cascading, partially unobservable
factors (ATC, crew scheduling, gate conflicts).  The project demonstrates
*scalable pipeline design* rather than state-of-the-art prediction accuracy.
The two-stage hurdle model meaningfully improves MAE, showing that treating
the delay distribution as a mixture of "no delay" and "delayed" is more
appropriate than direct regression.

---

## License

MIT

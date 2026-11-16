# spark_docker/ — Spark Cluster Infrastructure

Self-contained Docker Compose setup for running the flight delay pipeline
on a 3-node Spark standalone cluster.

## Prerequisites

- Docker Desktop running
- Spark binary: `spark-3.5.1-bin-without-hadoop/` in this folder
  (download once, not committed to git)
- `zscaler-ca.crt` for corporate networks (already included)

## Quick Start

```bash
cd spark_docker

# Build image + start cluster + run pipeline
./build.sh --run

# Or step by step:
./build.sh              # build image
./build.sh --up         # build + start cluster

# Submit a specific step
docker compose --profile driver run --rm driver \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /app/main.py --steps ingest prep features
```

## Folder Layout

```
spark_docker/
├── Dockerfile.spark                     # MCR-based, corporate-network friendly
├── docker-compose.yml                   # 1 master + 3 workers + driver
├── build.sh                             # Build helper (copies project source)
├── submit.sh                            # Job submission shortcut
├── zscaler-ca.crt                       # Zscaler root CA for HTTPS inspection
├── requirements.txt                     # (copied at build time)
├── spark-3.5.1-bin-without-hadoop/      # Spark binaries (not in git)
├── app_source/                          # Project source (copied at build time)
└── volumes/                             # Docker volume mounts
    ├── data/                            #   raw + processed + features
    ├── models/                          #   saved model artefacts
    ├── results/                         #   metrics CSVs + JSONs
    ├── logs/                            #   pipeline + download logs
    └── benchmarks_output/               #   benchmark results
```

## Web UIs

| URL | Description |
|-----|-------------|
| http://localhost:8080 | Spark Master (cluster overview) |
| http://localhost:8081 | Worker 1 |
| http://localhost:8082 | Worker 2 |
| http://localhost:8083 | Worker 3 |
| http://localhost:4040 | Spark App UI (active during job) |

## Notes

- The Dockerfile uses **MCR** (Microsoft Container Registry) base images
  to bypass Zscaler/CloudFront blocks on corporate networks.
- Spark binaries and Docker volumes are in `.gitignore` — only the config
  files are committed.

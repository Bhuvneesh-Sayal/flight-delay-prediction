#!/usr/bin/env bash
# =============================================================================
# spark_docker/build.sh
# =============================================================================
# Copies the project source into the Docker build context, builds the image,
# and optionally starts the cluster.
#
# Usage:
#   ./build.sh              # build image only
#   ./build.sh --up         # build + start cluster
#   ./build.sh --run        # build + start + submit main.py
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

echo "▶ Copying project source into Docker build context..."
rm -rf app_source
mkdir -p app_source

# Copy project files (not Docker infra)
cp "$PROJECT_ROOT/main.py"         app_source/
cp "$PROJECT_ROOT/pyproject.toml"  app_source/
cp "$PROJECT_ROOT/requirements.txt" app_source/
cp -r "$PROJECT_ROOT/config"       app_source/config
cp -r "$PROJECT_ROOT/src"          app_source/src
cp -r "$PROJECT_ROOT/app"          app_source/app 2>/dev/null || true
cp -r "$PROJECT_ROOT/tests"        app_source/tests 2>/dev/null || true

# Also copy requirements.txt into the Docker context root for the Dockerfile
cp "$PROJECT_ROOT/requirements.txt" .

echo "▶ Building Docker image..."
docker compose build

echo "✔ Image built: spark-local:3.5.1"

# Create volume dirs
mkdir -p volumes/{data,models,results,logs,benchmarks_output}

if [[ "${1:-}" == "--up" || "${1:-}" == "--run" ]]; then
    echo "▶ Starting cluster..."
    docker compose up -d
    echo "✔ Cluster running. Master UI: http://localhost:8080"
fi

if [[ "${1:-}" == "--run" ]]; then
    echo "▶ Submitting pipeline..."
    docker compose --profile driver run --rm driver \
        /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --deploy-mode client \
        --executor-memory 2g \
        --driver-memory 2g \
        /app/main.py "$@"
fi

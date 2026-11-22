#!/usr/bin/env bash
# =============================================================================
# docker/submit.sh
# =============================================================================
# Convenience wrapper to spark-submit main.py into the Docker Spark cluster.
#
# Usage (run from repo root):
#   ./docker/submit.sh                          # run all steps
#   ./docker/submit.sh --steps cluster_bench    # cluster benchmarks only
#   ./docker/submit.sh --steps ablation hurdle  # specific steps
#
# Prerequisites:
#   docker compose build           # build the image once
#   docker compose up -d           # start master + workers
#
# The driver container is started with --rm so it cleans up after the job.
# =============================================================================

set -euo pipefail

MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-2g}"
EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-3g}"
EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-2}"

# Pass any CLI arguments straight through to main.py
MAIN_ARGS="${*:-}"

# The Kafka source needs the Spark SQL Kafka connector on the classpath.
# Under spark-submit the JVM is already running when main.py builds its
# SparkSession, so spark.jars.packages set in code is ignored — the package
# must be passed here. Only added when the 'stream' step will run (it is
# downloaded from Maven Central on first use).
KAFKA_SPARK_PACKAGE="${KAFKA_SPARK_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0}"
PACKAGES_ARGS=()
if [[ -z "${MAIN_ARGS}" || " ${MAIN_ARGS} " == *" stream "* ]]; then
    PACKAGES_ARGS=(--packages "${KAFKA_SPARK_PACKAGE}")
fi

echo "=================================================="
echo " Flight Delay Prediction — Cluster Job Submission"
echo "=================================================="
echo " Master   : ${MASTER_URL}"
echo " Driver   : ${DRIVER_MEMORY} RAM"
echo " Executor : ${EXECUTOR_MEMORY} RAM × ${EXECUTOR_CORES} cores each"
echo " Args     : ${MAIN_ARGS:-<none — running all steps>}"
echo " Packages : ${PACKAGES_ARGS[1]:-<none>}"
echo "=================================================="
echo ""

docker compose run --rm \
    -e SPARK_MASTER_URL="${MASTER_URL}" \
    driver \
    /opt/spark/bin/spark-submit \
        --master "${MASTER_URL}" \
        --deploy-mode client \
        --driver-memory "${DRIVER_MEMORY}" \
        --executor-memory "${EXECUTOR_MEMORY}" \
        --executor-cores "${EXECUTOR_CORES}" \
        --conf spark.driver.host=spark-driver \
        --conf spark.driver.bindAddress=0.0.0.0 \
        --conf "spark.sql.shuffle.partitions=24" \
        --conf "spark.sql.adaptive.enabled=true" \
        ${PACKAGES_ARGS[@]+"${PACKAGES_ARGS[@]}"} \
        /app/main.py ${MAIN_ARGS}

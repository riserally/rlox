#!/bin/bash
set -euo pipefail
exec > /var/log/rlox-experiment.log 2>&1

echo "=== Installing Docker ==="
apt-get update && apt-get install -y docker.io docker-compose-v2

echo "=== Cloning rlox ==="
cd /tmp
git clone https://github.com/riserally/rlox.git
cd /tmp/rlox
mkdir -p results && chmod 777 results

RLOX_DIR=/tmp/rlox

echo "=== Building Docker image ==="
docker compose -f "${RLOX_DIR}/docker-compose.yml" build

echo "=== Running convergence benchmarks ==="
cd "${RLOX_DIR}"
docker compose run --rm benchmark-convergence

echo "=== Verifying results ==="
find "${RLOX_DIR}/results" -type f -ls

# Upload with retries
set +e
echo "=== Uploading results to GCS ==="
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DEST="gs://rkox-bench-results/convergence-${TIMESTAMP}"

for attempt in 1 2 3; do
  echo "Upload attempt ${attempt}..."
  gcloud storage cp -r "${RLOX_DIR}/results" "${DEST}/" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "Upload succeeded on attempt ${attempt}"
    gcloud storage ls -r "${DEST}/"
    break
  fi
  echo "Upload failed (rc=${RC}), retrying in 10s..."
  sleep 10
done

gcloud storage cp /var/log/rlox-experiment.log "${DEST}/experiment.log" 2>&1 || true

echo "=== Done — shutting down ==="
shutdown -h now

#!/bin/bash
set -euo pipefail
exec > /var/log/rlox-experiment.log 2>&1

echo "=== Installing Docker ==="
apt-get update && apt-get install -y docker.io docker-compose-v2

echo "=== Cloning rlox ==="
cd /tmp
git clone https://github.com/riserally/rlox.git
cd rlox
mkdir -p results && chmod 777 results

echo "=== Building Docker image ==="
docker compose build

echo "=== Running component benchmarks ==="
docker compose run --rm benchmark-components

echo "=== Verifying results ==="
find ./results -type f -ls

# Upload with retries — do not let failures kill the script
set +e
echo "=== Uploading results to GCS ==="
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DEST="gs://rkox-bench-results/components-${TIMESTAMP}"

for attempt in 1 2 3; do
  echo "Upload attempt ${attempt}..."
  gcloud storage cp -r ./results "${DEST}/" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "Upload succeeded on attempt ${attempt}"
    gcloud storage ls -r "${DEST}/"
    break
  fi
  echo "Upload failed (rc=${RC}), retrying in 10s..."
  sleep 10
done

# Also upload the full log
gcloud storage cp /var/log/rlox-experiment.log "${DEST}/experiment.log" 2>&1 || true

echo "=== Done — shutting down ==="
shutdown -h now

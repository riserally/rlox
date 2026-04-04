#!/bin/bash
# Check status of all GCP convergence experiments
# Usage: ./scripts/check-experiments.sh

set -euo pipefail
PROJECT="rkox-bench"
ZONE="us-central1-c"

echo "=== GCP Instances ==="
gcloud compute instances list --project="$PROJECT" --format="table(name,zone,machineType,status)" 2>&1

echo ""
for INSTANCE in $(gcloud compute instances list --project="$PROJECT" --filter="status=RUNNING" --format="value(name)" 2>/dev/null); do
    echo "=== $INSTANCE ==="

    # Try docker logs first (convergence instances)
    LOGS=$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --command="sudo docker logs --tail 3 \$(sudo docker ps -q 2>/dev/null) 2>/dev/null || tail -3 /var/log/rlox-*.log 2>/dev/null || echo 'No logs found'" 2>/dev/null || echo "SSH failed")
    echo "$LOGS"

    # Count results
    COUNT=$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --command="ls /tmp/rlox/results/convergence/*.json 2>/dev/null | wc -l || echo 0" 2>/dev/null || echo "?")
    echo "Results: $COUNT files"
    echo ""
done

echo "=== GCS Buckets ==="
gcloud storage ls gs://rkox-bench-results/ 2>&1 || echo "No GCS access"

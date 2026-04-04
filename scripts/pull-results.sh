#!/bin/bash
# Pull convergence results from a GCP instance
# Usage: ./scripts/pull-results.sh <instance-name> [output-dir]

set -euo pipefail
PROJECT="rkox-bench"
ZONE="us-central1-c"
INSTANCE="${1:?Usage: pull-results.sh <instance-name> [output-dir]}"
OUTPUT="${2:-results/$(echo $INSTANCE | sed 's/rlox-//')}"

mkdir -p "$OUTPUT"
echo "Pulling results from $INSTANCE to $OUTPUT/"
gcloud compute scp --recurse "$INSTANCE:/tmp/rlox/results/convergence/" "$OUTPUT/" \
    --zone="$ZONE" --project="$PROJECT" 2>&1

echo "Done. $(ls "$OUTPUT"/*.json 2>/dev/null | wc -l) files pulled."

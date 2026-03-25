#!/bin/bash
set -euo pipefail
exec > /var/log/rlox-experiment.log 2>&1

echo "=== Installing NVIDIA drivers ==="
apt-get update
apt-get install -y linux-headers-$(uname -r) pciutils
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-drivers

echo "=== Installing Docker + NVIDIA Container Toolkit ==="
apt-get install -y docker.io docker-compose-v2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== Verifying GPU ==="
nvidia-smi

echo "=== Cloning rlox ==="
cd /tmp
git clone https://github.com/riserally/rlox.git
cd rlox
mkdir -p results && chmod 777 results

echo "=== Building Docker image ==="
docker compose build

echo "=== Running GPU convergence benchmarks ==="
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm benchmark-convergence

echo "=== Verifying results ==="
find ./results -type f -ls

# Upload with retries
set +e
echo "=== Uploading results to GCS ==="
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DEST="gs://rkox-bench-results/convergence-gpu-${TIMESTAMP}"

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

gcloud storage cp /var/log/rlox-experiment.log "${DEST}/experiment.log" 2>&1 || true

echo "=== Done — shutting down ==="
shutdown -h now

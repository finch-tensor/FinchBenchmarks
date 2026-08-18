#!/bin/bash
set -e

FULL="${1:-0}"
THREADS="${2:-16}"

if [ "$FULL" = "0" ]; then
    DATASET="large_short"
else
    DATASET="large"
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_mttkrp_dense"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/data"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    -v "$SCRIPT_DIR/data:/repo/mttkrp/data" \
    "$IMAGE" bash -c "
cd /repo/mttkrp
if [ ! -d data ] || [ -z \"\$(ls -A data 2>/dev/null)\" ]; then
    bash get_mttkrp_data.sh
fi
julia -t $THREADS run_mttkrp.jl --dataset $DATASET --output results/mttkrp_results.json --ncpu $THREADS
cd results
poetry run python3 plot_mttkrp_results.py
"

docker cp "$CONTAINER_NAME:/repo/mttkrp/results/mttkrp_results.json" "$SCRIPT_DIR/results/mttkrp_results.json"
docker cp "$CONTAINER_NAME:/repo/mttkrp/results/mttkrp_speedup.png" "$SCRIPT_DIR/results/mttkrp_speedup.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

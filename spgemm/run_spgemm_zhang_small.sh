#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="finch_spgemm_zhang_small"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/spgemm
julia -t $THREADS run_spgemm.jl --dataset zhang_small --kernels full --output results/zhang_small_results_full.json
cd results
poetry run python3 plot_zhang_speedup_full.py
"

docker cp "$CONTAINER_NAME:/repo/spgemm/results/zhang_small_results_full.json" "$SCRIPT_DIR/results/zhang_small_results_full.json"
docker cp "$CONTAINER_NAME:/repo/spgemm/results/zhang_speedup_full.png" "$SCRIPT_DIR/results/zhang_speedup_full.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_spmspv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/spmspv
julia -t $THREADS run_spmspv.jl --dataset snap_largest --output results/spmspv_results.json
cd results
poetry run python3 plot.py
"

docker cp "$CONTAINER_NAME:/repo/spmspv/results/spmspv_results.json" "$SCRIPT_DIR/results/spmspv_results.json"
docker cp "$CONTAINER_NAME:/repo/spmspv/results/spmspv_speedup.png" "$SCRIPT_DIR/results/spmspv_speedup.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

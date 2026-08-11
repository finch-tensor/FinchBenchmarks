#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_hist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/structured/histogram
julia -t $THREADS run_hist.jl --dataset image --ncpu $THREADS
cd results
poetry run python3 plot_hist_strong_scaling.py
"

docker cp "$CONTAINER_NAME:/repo/structured/histogram/results/hist_${THREADS}_threads.json" "$SCRIPT_DIR/results/hist_${THREADS}_threads.json"
docker cp "$CONTAINER_NAME:/repo/structured/histogram/results/hist_strong_scaling.png" "$SCRIPT_DIR/results/hist_strong_scaling.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_struct_add"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/structured/add
julia -t $THREADS run_add.jl --dataset image --ncpu $THREADS
cd results
poetry run python3 plot_add_strong_scaling.py
"

docker cp "$CONTAINER_NAME:/repo/structured/add/results/add_${THREADS}_threads.json" "$SCRIPT_DIR/results/add_${THREADS}_threads.json"
docker cp "$CONTAINER_NAME:/repo/structured/add/results/add_strong_scaling.png" "$SCRIPT_DIR/results/add_strong_scaling.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

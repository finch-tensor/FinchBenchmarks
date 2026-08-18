#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_hadamard_sparse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/hadamard
julia -t $THREADS run_hadamard.jl --dataset sparse --output results/hadamard_sparse_results.json --ncpu $THREADS
cd results
poetry run python3 plot.py sparse
"

docker cp "$CONTAINER_NAME:/repo/hadamard/results/hadamard_sparse_results.json" "$SCRIPT_DIR/results/hadamard_sparse_results.json"
docker cp "$CONTAINER_NAME:/repo/hadamard/results/hadamard_sparse_speedup.png" "$SCRIPT_DIR/results/hadamard_sparse_speedup.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

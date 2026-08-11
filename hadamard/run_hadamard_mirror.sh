#!/bin/bash
set -e

THREADS="${1:-16}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_hadamard_mirror"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e OMP_NUM_THREADS \
    -e MKL_NUM_THREADS \
    "$IMAGE" bash -c "
cd /repo/hadamard
julia -t $THREADS run_hadamard.jl --dataset mirror --output results/hadamard_mirror_results.json --ncpu $THREADS
cd results
poetry run python3 plot.py mirror
"

docker cp "$CONTAINER_NAME:/repo/hadamard/results/hadamard_mirror_results.json" "$SCRIPT_DIR/results/hadamard_mirror_results.json"
docker cp "$CONTAINER_NAME:/repo/hadamard/results/hadamard_mirror_speedup.png" "$SCRIPT_DIR/results/hadamard_mirror_speedup.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

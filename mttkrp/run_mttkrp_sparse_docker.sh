#!/bin/bash
set -e

MAX_THREADS="${1:-16}"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_mttkrp_sparse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e MAX_THREADS="$MAX_THREADS" \
    "$IMAGE" bash -c '
cd /repo/mttkrp
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_mttkrp.jl --dataset sparse --output "results/mttkrp_sparse_t${t}.json" --ncpu "$t"
    t=$((t * 2))
done
cd results
poetry run python3 plot_mttkrp_strong_scaling.py
'

t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    docker cp "$CONTAINER_NAME:/repo/mttkrp/results/mttkrp_sparse_t${t}.json" "$SCRIPT_DIR/results/mttkrp_sparse_t${t}.json"
    t=$((t * 2))
done
docker cp "$CONTAINER_NAME:/repo/mttkrp/results/mttkrp_strong_scaling.png" "$SCRIPT_DIR/results/mttkrp_strong_scaling.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

#!/bin/bash
set -e

FULL="${1:-0}"
MAX_THREADS="${2:-16}"

if [ "$FULL" = "0" ]; then
    DATASET="sparse_short"
else
    DATASET="sparse"
fi

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_mttkrp_sparse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/data"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e MAX_THREADS="$MAX_THREADS" \
    -e DATASET="$DATASET" \
    -v "$SCRIPT_DIR/data:/repo/mttkrp/data" \
    "$IMAGE" bash -c '
cd /repo/mttkrp
if [ ! -d data ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    bash get_mttkrp_data.sh
fi
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_mttkrp.jl --dataset "$DATASET" --output "results/mttkrp_sparse_t${t}.json" --ncpu "$t"
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

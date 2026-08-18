#!/bin/bash
set -e

MAX_THREADS="${1:-16}"

IMAGE="wingspan:cgo27"
CONTAINER_NAME="wingspan_spgemm_strong_scale"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" \
    -e MAX_THREADS="$MAX_THREADS" \
    "$IMAGE" bash -c '
cd /repo/spgemm
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    export OMP_NUM_THREADS="$t"
    export MKL_NUM_THREADS="$t"
    julia -t "$t" run_spgemm.jl --dataset w5 --kernels fast --output "results/spgemm_scale_threads_${t}.json"
    t=$((t * 2))
done
cd results
poetry run python3 plot_strong_scaling.py
'

t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    docker cp "$CONTAINER_NAME:/repo/spgemm/results/spgemm_scale_threads_${t}.json" "$SCRIPT_DIR/results/spgemm_scale_threads_${t}.json"
    t=$((t * 2))
done
docker cp "$CONTAINER_NAME:/repo/spgemm/results/strong_scaling.png" "$SCRIPT_DIR/results/strong_scaling.png"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1

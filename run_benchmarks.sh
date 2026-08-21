#!/bin/bash
#
# Runs every benchmarking script in the repo.
#
# Usage: ./run_benchmarks.sh [FULL] [THREADS]
#   FULL    - passed through to the mttkrp scripts (0 or 1, default 0),
#             selecting the short or full-size mttkrp datasets.
#   THREADS - max thread count passed to every script (default 16).

FULL="${1:-0}"
THREADS="${2:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MTTKRP_SCRIPTS=(
    "mttkrp/run_mttkrp_dense.sh"
    "mttkrp/run_mttkrp_sparse.sh"
)

THREADS_ONLY_SCRIPTS=(
    "spgemm/run_spgemm_strong_scale.sh"
    "spgemm/run_spgemm_weak_scale.sh"
    "spgemm/run_spgemm_zhang_large.sh"
    "spgemm/run_spgemm_zhang_small.sh"
    "spadd/run_spadd_mirror.sh"
    "spadd/run_spadd_sparse.sh"
    "hadamard/run_hadamard_mirror.sh"
    "hadamard/run_hadamard_sparse.sh"
    "spmspv/run_spmspv.sh"
    "structured/add/run_struct_add.sh"
    "structured/histogram/run_hist.sh"
)

FAILED=()

run_script() {
    local rel_path="$1"
    shift
    echo "===== Running $rel_path $* ====="
    if ! (cd "$SCRIPT_DIR" && bash "$rel_path" "$@"); then
        echo "===== FAILED: $rel_path =====" >&2
        FAILED+=("$rel_path")
    fi
}

for script in "${MTTKRP_SCRIPTS[@]}"; do
    run_script "$script" "$FULL" "$THREADS"
done

for script in "${THREADS_ONLY_SCRIPTS[@]}"; do
    run_script "$script" "$THREADS"
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All benchmarks completed successfully."
else
    echo "The following benchmarks failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi

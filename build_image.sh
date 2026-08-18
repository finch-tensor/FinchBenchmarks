#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"
DOCKER_REF="${DOCKER_REF:-docker://agushin/wingspan-cgo27-artifact:v1}"

if [ -f "$SIF_IMAGE" ]; then
    if [ "${FORCE_PULL:-0}" = "1" ]; then
        rm -f "$SIF_IMAGE"
    fi
fi

if [ ! -f "$SIF_IMAGE" ]; then
    echo "Pulling $DOCKER_REF -> $SIF_IMAGE"
    apptainer pull "$SIF_IMAGE" "$DOCKER_REF"
fi

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    "$SIF_IMAGE" bash -c '
cd /repo
make all
'

echo "Done."

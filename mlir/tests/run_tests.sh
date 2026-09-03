#!/usr/bin/env bash

LLVM_BUILD="/opt/homebrew/opt/llvm"

"$LLVM_BUILD/bin/mlir-opt" read_mtx.mlir \
    --sparsifier="enable-runtime-library=true" \
| "$LLVM_BUILD/bin/mlir-runner" \
    -e main \
    --entry-point-result=void \
    --shared-libs="$LLVM_BUILD/lib/libmlir_c_runner_utils.dylib"
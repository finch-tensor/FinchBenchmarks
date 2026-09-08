#!/usr/bin/env bash

LLVM="/opt/homebrew/opt/llvm"
SPADD_DIR="/Users/pau/Developer/finch/fbench/spadd"
ROOT="/Users/pau/Developer/finch/fbench"
OUT="$SPADD_DIR/build"
BIN="$SPADD_DIR/spadd_mlir"

mkdir -p "$OUT"

for f in load_mtx spadd print_csr; do
  echo "compiling mlir/$f.mlir"
  "$LLVM/bin/mlir-opt" "$ROOT/mlir/$f.mlir" --sparsifier="enable-runtime-library=true" \
    | "$LLVM/bin/mlir-translate" --mlir-to-llvmir \
    | "$LLVM/bin/llc" -filetype=obj -o "$OUT/$f.o"
done

echo "linking $BIN"
clang++ -std=c++17 "$SPADD_DIR/mlir_impl.cpp" \
  "$OUT/load_mtx.o" "$OUT/spadd.o" "$OUT/print_csr.o" \
  -L"$LLVM/lib" -lmlir_c_runner_utils -Wl,-rpath,"$LLVM/lib" \
  -o "$BIN"

julia run_spadd.jl --dataset mirror -m mlir_impl --ncpu 1 -o results/spadd_mlir.json
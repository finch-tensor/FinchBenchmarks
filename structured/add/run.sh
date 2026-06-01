#!/bin/bash

echo "Building opencv kernel..."
make -C "$(dirname "$0")" clean; make -C "$(dirname "$0")" || { echo "Build failed"; exit 1; }

for (( t=1 ; t<=$1 ; t*=2));
do
    echo "Running run_add.jl with $t threads"
    julia --threads=$t run_add.jl --ncpu $t
done

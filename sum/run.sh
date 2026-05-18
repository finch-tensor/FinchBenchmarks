#!/bin/bash

echo "Building TACO, Eigen, MKL kernel..."
make -C "$(dirname "$0")" clean; make -C "$(dirname "$0")" || { echo "Build failed"; exit 1; }

for (( t=1 ; t<=$1 ; t*=2));
do
    echo "Running run_sum.jl with $t threads"
    julia --threads=$t run_sum.jl --ncpu $t
done

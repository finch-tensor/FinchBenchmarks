#!/bin/bash

for (( t=1 ; t<=$1 ; t*=2));
do
    echo "Running run_hist.jl with $t threads"
    julia --threads=$t run_hist.jl --ncpu $t
done

#!/bin/bash

mkdir -p "results"

for (( t=1 ; t<=$1; t++));
do
	julia -t "$t" run_spmv.jl -o "results/spmv_threads_${t}.json"
done

jq -s 'add' "results/spmv_threads_*.json" > "results/spmv_results.json"

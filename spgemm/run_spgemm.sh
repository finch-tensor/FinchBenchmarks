#!/bin/bash

mkdir -p "results"

for (( t=1 ; t<=$1; t++));
do
	julia -t "$t" run_spgemm.jl -o "results/spgemm_threads_${t}.json"
done

jq -s add results/spgemm_threads_*.json > results/spgemm_results.json

#!/bin/bash

for (( t=1 ; t<=$1; t++));
do
	julia -t "$t" run_spmv.jl -o "spmv_threads_${t}.json"
done

jq -s 'add' spmv_threads_*.json > spmv_results.json

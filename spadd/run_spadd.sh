#!/bin/bash

for (( t=1 ; t<=$1; t++));
do
	julia -t "$t" run_spadd.jl -o "spadd_threads_${t}.json"
done

jq -s 'add' spadd_threads_*.json > spadd_results.json

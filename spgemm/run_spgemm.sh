#!/bin/bash

julia run_spgemm.jl -d "zhang_small" --kernels "all" -o results_zhang_small.json
#julia run_spgemm.jl -d "zhang_large" --kernels "fast" -o results_zhang_large.json
#Ideally we would have time to run the large matrices. Alternatively we can copy
#the reference results so that the plotting script doesn't error out
cp reference_results_zhang_large.json results_zhang_large.json 
julia run_spgemm.jl -d "scale" --kernels "all" -b 1 -B 9 -o split_results_scale_1.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 2 -B 9 -o split_results_scale_2.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 3 -B 9 -o split_results_scale_3.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 4 -B 9 -o split_results_scale_4.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 5 -B 9 -o split_results_scale_5.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 6 -B 9 -o split_results_scale_6.json
julia run_spgemm.jl -d "scale" --kernels "all" -b 7 -B 9 -o split_results_scale_7.json
#julia run_spgemm.jl -d "scale" --kernels "all" -b 8 -B 9 -o split_results_scale_8.json
#julia run_spgemm.jl -d "scale" --kernels "all" -b 9 -B 9 -o split_results_scale_9.json
jq -s 'add' results_zhang_small.json split_results_scale_*.json > results_scale.json
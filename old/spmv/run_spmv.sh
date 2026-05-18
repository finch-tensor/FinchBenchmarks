#!/bin/bash

julia run_spmv.jl -o split_results.json
source ../deps/intel/setvars.sh; PYTHONPATH=../deps/cora/python/ poetry run python trmv_cora.py --m 1024 --n 1
jq -s 'add' split_results.json spmv_results_cora.json > spmv_results.json
#!/bin/bash
source ../deps/intel/setvars.sh; PYTHONPATH=../deps/cora/python/ poetry run python trmv_cora.py --m 1024 --n 1

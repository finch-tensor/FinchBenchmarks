#!/bin/bash
julia --project=. -e '
using Pkg;
Pkg.instantiate(); 
Pkg.precompile()
'

poetry install --no-root

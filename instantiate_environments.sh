#!/bin/bash
julia --project=. -e '
using Pkg; 
Pkg.develop(path="Finch.jl")
Pkg.instantiate(); 
Pkg.precompile()
'

poetry install --no-root

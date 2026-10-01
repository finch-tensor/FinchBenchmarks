#!/usr/bin/env julia
# Runs run_spmspv.jl's benchmark body under whatever --project was passed to
# this process, instead of run_spmspv.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for everything but birdseed: wingspan lives in its own Finch checkout
# (envs/wingspan). Invoke as
#   julia --project=envs/wingspan -t N,0 spmspv/run_spmspv_wingspan.jl -m wingspan-bytemap-static ...
include(joinpath(@__DIR__, "run_spmspv.jl"))

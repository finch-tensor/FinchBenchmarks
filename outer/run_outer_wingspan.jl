#!/usr/bin/env julia
# Runs run_outer.jl's benchmark body under whatever --project was passed to
# this process, instead of run_outer.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for everything but birdseed: wingspan lives in its own Finch checkout
# (envs/wingspan). Invoke as
#   julia --project=envs/wingspan -t N,0 outer/run_outer_wingspan.jl -m mkl -m wingspan_outer
include(joinpath(@__DIR__, "run_outer.jl"))

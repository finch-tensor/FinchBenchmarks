#!/usr/bin/env julia
# Runs run_spadd.jl's benchmark body under whatever --project was passed to
# this process, instead of run_spadd.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for everything but birdseed: wingspan lives in its own Finch checkout
# (envs/wingspan). Invoke as
#   julia --project=envs/wingspan -t N spadd/run_spadd_wingspan.jl -m wingspan_spadd ...
include(joinpath(@__DIR__, "run_spadd.jl"))

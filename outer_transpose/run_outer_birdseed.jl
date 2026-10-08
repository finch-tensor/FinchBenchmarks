#!/usr/bin/env julia
# Runs run_outer.jl's benchmark body under whatever --project was passed to
# this process, instead of run_outer.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for the birdseed baseline: birdseed lives in its own Finch checkout (envs/birdseed),
# which is the same "Finch" package/UUID as the main env, so it has to run in
# its own process. Invoke as
#   julia --project=envs/birdseed -t N,0 outer_transpose/run_outer_birdseed.jl -m mkl_0 ... -m mkl_7 -m birdseed_outer
include(joinpath(@__DIR__, "run_outer.jl"))

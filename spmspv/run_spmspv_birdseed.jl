#!/usr/bin/env julia
# Runs run_spmspv.jl's benchmark body under whatever --project was passed to
# this process, instead of run_spmspv.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for the birdseed baseline: birdseed lives in its own Finch checkout (envs/birdseed),
# which is the same "Finch" package/UUID as the main env, so it has to run in
# its own process. Invoke as
#   julia --project=envs/birdseed -t N spmspv/run_spmspv_birdseed.jl -m coalesce_static -m coalesce_dynamic ...
include(joinpath(@__DIR__, "run_spmspv.jl"))

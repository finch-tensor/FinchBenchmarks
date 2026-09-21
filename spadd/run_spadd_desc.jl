#!/usr/bin/env julia
# Runs run_spadd.jl's benchmark body under whatever --project was passed to
# this process, instead of run_spadd.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project stays active.
#
# Used for the desc baseline: desc lives in its own Finch checkout (envs/desc),
# which is the same "Finch" package/UUID as the main env, so it has to run in
# its own process. Invoke as
#   julia --project=envs/desc -t N spadd/run_spadd_desc.jl -m desc_spadd ...
include(joinpath(@__DIR__, "run_spadd.jl"))

#!/usr/bin/env julia
# Runs run_spadd.jl's benchmark body under whatever --project was passed to
# this process, instead of run_spadd.jl's own `Pkg.activate(repo_root)` (which
# only fires when it detects itself as the top-level script). `include`ing it
# here keeps that check false, so the caller's --project (a Finch worktree
# env) stays active.
include(joinpath(@__DIR__, "run_spadd.jl"))

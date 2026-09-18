#!/usr/bin/env julia
# Runs run_spmspv.jl once per Finch checkout (envs/wingspan, envs/desc) in a
# separate `julia --project` subprocess -- both checkouts are the "Finch"
# package under the same UUID, so they can't be loaded in one process -- then
# merges the two result JSONs, tagging each row with which version produced it.
#
# All CLI args (--method, --dataset, --accuracy-check) are forwarded to
# run_spmspv.jl as-is, so restrict to Finch-backed methods with e.g.
# `-m coalesce_static` or `-m coalesce_dynamic` -- there's no point re-running
# the eigen/graphblas methods twice, they don't depend on Finch. Unlike SpAdd,
# run_spmspv.jl has no --ncpu; it uses Threads.nthreads(), so --threads is
# passed to each subprocess as `julia -t`.
using ArgParse
using JSON

s = ArgParseSettings("Compare Finch versions (wingspan-arxiv-v0 vs desc) on SpMSpV.")
@add_arg_table! s begin
    "--threads", "-t"
    arg_type = Int
    default = 16
    help = "number of Julia threads for each run"
    "--output", "-o"
    arg_type = String
    default = "results/spmspv_compare.json"
    "--dataset", "-d"
    arg_type = String
    "--method", "-m"
    arg_type = String
    "--accuracy-check", "-a"
    action = :store_true
end
parsed_args = parse_args(ARGS, s)

repo_root = dirname(@__DIR__)
worker = joinpath(@__DIR__, "_run_spmspv_worker.jl")

versions = [
    "wingspan" => joinpath(repo_root, "envs", "wingspan"),
    "desc" => joinpath(repo_root, "envs", "desc"),
]

forwarded = String[]
for (flag, key) in ("--dataset" => "dataset", "--method" => "method")
    if !isnothing(parsed_args[key])
        push!(forwarded, flag, string(parsed_args[key]))
    end
end
parsed_args["accuracy-check"] && push!(forwarded, "--accuracy-check")

results = []
mktempdir() do tmp
    for (version, env) in versions
        out = joinpath(tmp, "$version.json")
        cmd = `julia --project=$env -t $(parsed_args["threads"]) $worker $forwarded -o $out`
        @info "Running SpMSpV with Finch=$version" cmd
        run(cmd)
        for row in JSON.parsefile(out)
            row["version"] = version
            push!(results, row)
        end
    end
end

mkpath(dirname(parsed_args["output"]))
write(parsed_args["output"], JSON.json(results, 4))

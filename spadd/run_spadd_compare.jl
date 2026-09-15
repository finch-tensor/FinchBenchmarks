#!/usr/bin/env julia
# Runs run_spadd.jl once per Finch checkout (envs/wingspan, envs/desc) in a
# separate `julia --project` subprocess -- both checkouts are the "Finch"
# package under the same UUID, so they can't be loaded in one process -- then
# merges the two result JSONs, tagging each row with which version produced it.
#
# All CLI args (--method, --dataset, --ncpu, --accuracy-check) are forwarded
# to run_spadd.jl as-is, so restrict to Finch-backed methods with e.g.
# `-m serial_default_implementation` or `-m shard_implementation` -- there's
# no point re-running the eigen/mkl/graphblas methods twice, they don't
# depend on Finch.
using ArgParse
using JSON

s = ArgParseSettings("Compare Finch versions (wingspan-arxiv-v0 vs desc) on SpAdd.")
@add_arg_table! s begin
    "--ncpu"
    arg_type = Int
    "--output", "-o"
    arg_type = String
    default = "results/spadd_compare.json"
    "--dataset", "-d"
    arg_type = String
    "--method", "-m"
    arg_type = String
    "--accuracy-check", "-a"
    action = :store_true
end
parsed_args = parse_args(ARGS, s)

repo_root = dirname(@__DIR__)
worker = joinpath(@__DIR__, "_run_spadd_worker.jl")

versions = [
    "wingspan" => joinpath(repo_root, "envs", "wingspan"),
    "desc" => joinpath(repo_root, "envs", "desc"),
]

forwarded = String[]
for (flag, key) in ("--ncpu" => "ncpu", "--dataset" => "dataset", "--method" => "method")
    if !isnothing(parsed_args[key])
        push!(forwarded, flag, string(parsed_args[key]))
    end
end
parsed_args["accuracy-check"] && push!(forwarded, "--accuracy-check")

results = []
mktempdir() do tmp
    for (version, env) in versions
        out = joinpath(tmp, "$version.json")
        cmd = `julia --project=$env $worker $forwarded -o $out`
        @info "Running SpAdd with Finch=$version" cmd
        run(cmd)
        for row in JSON.parsefile(out)
            row["version"] = version
            push!(results, row)
        end
    end
end

mkpath(dirname(parsed_args["output"]))
write(parsed_args["output"], JSON.json(results, 4))

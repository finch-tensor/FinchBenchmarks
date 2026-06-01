#!/usr/bin/env julia
# using Base: nothing_sentinel
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using FileIO
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Images

# Parsing Arguments
s = ArgParseSettings("Run Structured Add Experiments.")
@add_arg_table! s begin
   "--ncpu"
    help = "number of CPUs"
    arg_type = Int
    "--output", "-o"
    arg_type = String
    help = "output file path"
    "--dataset", "-d"
    arg_type = String
    help = "dataset keyword"
    "--method", "-m"
    arg_type = String
    help = "method keyword"
    "--accuracy-check", "-a"
    action = :store_true
    help = "check method accuracy"
end
parsed_args = parse_args(ARGS, s)

# Mapping from dataset types to datasets
datasets = Dict(
    "image" => [
        "highly_compressed/aviva.songs24.de.jpg",
    ],
)

include("coalesce_impl.jl")


methods = OrderedDict(
    "coalesce_impl" => coalesce_impl,
)

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        # Get relevant matrix
        if dataset == "image"
            A_orig = load("../" * mtx)
            m, n = size(A_orig)
            A = fill((0, 0, 0), m, n)

            for i in 1:m
                for j in 1:n
                    c = A_orig[i, j]

                    A[i, j] = (
                        round(Int, 255 * Float64(c.r)),
                        round(Int, 255 * Float64(c.g)),
                        round(Int, 255 * Float64(c.b))
                    )
                end
            end

        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the sum
                coalesce_impl_result = coalesce_impl(A, ncpu)

                m, n, p = size(coalesce_impl_result.hist)
                expected = coalesce_impl_result.hist
                for i in 1:m
                    for j in 1:n
                        for k in 1:p
                            @assert isapprox(result.hist[i,j,k], expected[i,j,k]) "Incorrect result for $key at ($i,$j,$k): got $(result.hist[i,j,k]), expected $(expected[i,j,k])"
                        end
                    end
                end
            end

            # Write result
            time = result.time
            @info "result for $key on $mtx" time
            push!(results, OrderedDict(
                "time" => time,
                "n_threads" => Threads.nthreads(),
                "method" => key,
                "dataset" => dataset,
                "matrix" => mtx,
            ))

            if isnothing(parsed_args["output"])
                write("results/sum_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
            else
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end

results = []
if isnothing(parsed_args["dataset"])
    for (dataset, mtxs) in datasets
        calculate_results(dataset, mtxs, results)
    end
else
    dataset = parsed_args["dataset"]
    mtxs = datasets[dataset]
    calculate_results(dataset, mtxs, results)
end



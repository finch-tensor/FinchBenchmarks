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
        ("highly_compressed/benjamin.duboc.free.fr.jpg",
            "highly_compressed/guitars.com.jpg"),
        
    ],
)

include("shard_impl.jl")


methods = OrderedDict(
    "shard_impl" => shard_impl,
)

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for (a_path, b_path) in mtxs
        # Get relevant matrix
        if dataset == "image"
            A_orig = load("../" * a_path)
            m, n = size(A_orig)
            A = zeros(UInt8, m, n)

            for i in 1:m
                for j in 1:n
                    c = A_orig[i, j]

                    A[i, j] = round(UInt8, (c.r * 0.299 + c.g * 0.587 + c.b * 0.114) * 255)
                end
            end

            B_orig = load("../" * b_path)
            B = zeros(UInt8, m, n)

            for i in 1:m
                for j in 1:n
                    c = B_orig[i, j]

                    B[i, j] = round(UInt8, (c.r * 0.299 + c.g * 0.587 + c.b * 0.114) * 255)
                end
            end

        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, B, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the sum
                shard_impl_result = shard_impl(A, B, ncpu)

                m, n = size(shard_impl_result.C)
                expected = shard_impl_result.C
                for i in 1:m
                    for j in 1:n
                        @assert isapprox(result.C[i,j], expected[i,j]) "Incorrect result for $key at ($i,$j): got $(result.C[i,j]), expected $(expected[i,j])"
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
                "matrix" => "$(a_path) + $(b_path)",
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

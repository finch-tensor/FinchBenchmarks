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
        "very_highly_compressed/www.abalip.com.jpg",
        "very_highly_compressed/www.carmelmusic.com.jpg",
        "very_highly_compressed/www.claudiozappi.it.jpg",
        "very_highly_compressed/www.duo-thais.com.jpg",
        "very_highly_compressed/www.handball-riehen.ch.jpg",
    ],
    # "uniform" => [
    #     OrderedDict("size" => 1_000, "sparsity" => 0.1),
    #     OrderedDict("size" => 1_000, "sparsity" => 0.01),
    #     OrderedDict("size" => 1_000, "sparsity" => 0.001),
    #     OrderedDict("size" => 1_000, "sparsity" => 0.0001),
    #     OrderedDict("size" => 2_000, "sparsity" => 0.00001),
    #     OrderedDict("size" => 5_000, "sparsity" => 0.00001),
    #     OrderedDict("size" => 10_000, "sparsity" => 0.00001),
    #     OrderedDict("size" => 20_000, "sparsity" => 0.00001),
    #     OrderedDict("size" => 50_000, "sparsity" => 0.00001),
    #     OrderedDict("size" => 100_000, "sparsity" => 0.00001),

    # ],
)

include("coalesce_impl.jl")
include("cv_impl.jl")


methods = OrderedDict(
    "coalesce_impl" => coalesce_impl,
    # "cv_impl" => hist_cv_impl,
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
            A = fill((UInt8(0), UInt8(0), UInt8(0)), m, n)

            for i in 1:m
                for j in 1:n
                    c = A_orig[i, j]

                    A[i,j] = (UInt8(c.r * 255), UInt8(c.g * 255), UInt8(c.b * 255))
                end
            end
        elseif dataset == "uniform"
            R = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            G = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            B = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            A = fill((UInt8(0), UInt8(0), UInt8(0)), mtx["size"], mtx["size"])
            for i in 1:mtx["size"]
                for j in 1:mtx["size"]
                    A[i, j] = (R[i, j], G[i, j], B[i, j])
                end
            end

        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the hist
                coalesce_impl_result = coalesce_impl(A, ncpu)

                m, n, p = size(coalesce_impl_result.hist)
                expected = coalesce_impl_result.hist
                for i in 1:m
                    @assert isapprox(result.hist[i], expected[i]) "Incorrect result for $key at ($i): got $(result.hist[i]), expected $(expected[i])"
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
                write("results/hist_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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

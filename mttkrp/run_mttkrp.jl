#!/usr/bin/env julia
# using Base: nothing_sentinel
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Random
using SparseArrays

Random.seed!(1234)

# Parsing Arguments
s = ArgParseSettings("Run Parallel MTTKRP Experiments.")
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
    "uniform" => [
        OrderedDict("size" => 1_000, "sparsity" => 1e-5),
    ],
    # need to manually download from http://frostt.io/tensors/
    "large" => [
        "data/nell-2.tns",
        "data/1998DARPA.tns",
        "data/fb-m.tns",
        "data/nell-1.tns",
     ],
     "large_short" => [
        "data/nell-2.tns",
        "data/1998DARPA.tns",
     ],
     "sparse" => [
        "data/fb-m.tns",
        "data/nell-2.tns",
        "data/1998DARPA.tns",
        "data/nell-1.tns",
     ],
    "sparse_short" => [
        "data/nell-2.tns",
        "data/1998DARPA.tns",
     ]
)

dataset_kind(name) = replace(name, r"_short$" => "")

# Mapping from method keywords to methods
include("taco_impl.jl")

isnothing(parsed_args["dataset"]) && error("--dataset is required")

if dataset_kind(parsed_args["dataset"]) == "sparse"
    include("finch_impl_sparse.jl")
    methods = OrderedDict(
        "finch_impl" => finch_impl,
    )
else
    include("finch_impl_dense.jl")
    methods = OrderedDict(
        "taco_impl" => taco_impl,
        "finch_impl" => finch_impl,
    )
end

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    kind = dataset_kind(dataset)
    for mtx in mtxs
        if kind == "uniform"
            B = fsprand(mtx["size"], mtx["size"], mtx["size"], mtx["sparsity"])
            C = rand(mtx["size"], 32)
            D = rand(mtx["size"], 32)
        elseif kind == "large"
            B = TensorMarket.tnsread(mtx)
	        B = fsparse(B[1]..., B[2])
            C = rand(size(B)[2], 16)
            D = rand(size(B)[3], 16)
        elseif kind == "sparse"
            B = TensorMarket.tnsread(mtx)
            if mtx == "data/fb-m.tns"
                B = fsparse(B[1][3], B[1][2], B[1][1], B[2])
            else
                B = fsparse(B[1]..., B[2])
            end
            C = sprand(size(B)[2], 16, 0.01)
            D = sprand(size(B)[3], 16, 0.01)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
	    @info "starting test on $key"
            result = method(B, C, D, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the sum
                finch_impl_result = finch_impl(B, C, D, ncpu)

                rtol = 1e-5
                for i in 1:size(result.A)[1]
                    for j in 1:size(result.A)[2]
                        @assert isapprox(result.A[i, j], finch_impl_result.A[i, j], rtol=rtol) """
                        Incorrect result for $key at index ($i, $j): got $(result.A[i, j]), expected $(finch_impl_result.A[i, j])
                        relative error = $(abs(result.A[i, j] - finch_impl_result.A[i, j]) / abs(finch_impl_result.A[i, j]))
                        """
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
                write("results/mttkrp_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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

#!/usr/bin/env julia
# using Base: nothing_sentinel
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Random

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
    # "nell" => [
    #     "data/nell-2.tns",
    # ],
)

# Mapping from method keywords to methods
include("finch_impl.jl")
include("taco_impl.jl")
include("eigen_impl.jl")
include("mkl_impl.jl")


methods = OrderedDict(
    "finch_impl" => finch_impl,
    "taco_impl" => taco_impl,
    "eigen_impl" => eigen_impl,
    "mkl_impl" => mkl_impl,
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
        if dataset == "uniform"
            B = fsprand(mtx["size"], mtx["size"], mtx["size"], mtx["sparsity"])
            C = rand(mtx["size"], 32)
            D = rand(mtx["size"], 32)
        elseif dataset == "nell"
            B = TensorMarket.tnsread(mtx) # may have to look for a sparser matrix?
            C = rand(size(B)[2], 32)
            D = rand(size(B)[3], 32)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
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

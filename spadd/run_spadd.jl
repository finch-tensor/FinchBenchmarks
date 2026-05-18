using Base: nothing_sentinel
#!/usr/bin/env julia
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
s = ArgParseSettings("Run Parallel SpAdd Experiments.")
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
    # all 1e-3 sparsity
    "diff_sizes" => [
        "FIDAP/ex27", # 1k x 1k
        "ND/nd3k", # 10k x 10k
        "DIMACS10/G_n_pin_pout", # 100k x 100k
    ],
    # all 10k x 10k
    "diff_sparsity" => [
        "Pajek/California", # 1e-5
        "Nasa/shuttle_eddy", # 1e-4
        "Nemeth/nemeth20", # 1e-3 (may have too many nnz)
    ],
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
# include("parallel_col_separate_sparselist_results.jl")
# include("separated_memory_concatenate_results.jl")
include("shard_implementation.jl")
# include("taco_impl.jl")
include("eigen_impl.jl")
include("mkl_impl.jl")
#include("graphBLAS_impl.jl")

methods = OrderedDict(
    "serial_default_implementation" => serial_default_implementation_add,
    # "parallel_col_separate_sparselist_results" => parallel_col_separate_sparselist_results_add,
    # "separated_memory_concatenate_results" => separated_memory_concatenate_results_add,
    "shard_implementation" => shard_add,
    # "taco_impl" => taco_impl,
    "eigen_impl" => eigen_impl,
    "mkl_impl" => mkl_impl,
    # "graphblas_impl" => graphblas_impl,
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
        if dataset == "diff_sizes" || dataset == "diff_sparsity"
            A = matrixdepot(mtx)
            row_permutation = randperm(size(A, 1))
            col_permutation = randperm(size(A, 2))
            B = A[row_permutation, col_permutation]
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, B, ncpu)

            if parsed_args["accuracy-check"]
                serial_default_implementation_result = serial_default_implementation_add(A, B, ncpu)
                expected = serial_default_implementation_result.C
                m, n = size(expected)
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
                "matrix" => mtx,
            ))

            if isnothing(parsed_args["output"])
                write("results/spadd_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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



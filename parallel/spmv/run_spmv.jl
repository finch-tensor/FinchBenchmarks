using Base: nothing_sentinel
#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using LinearAlgebra

using ThreadPinning
pinthreads(:cores)

# Parsing Arguments
s = ArgParseSettings("Run Parallel SpMV Experiments.")
@add_arg_table! s begin
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
        OrderedDict("size" => 2^10, "sparsity" => 0.1),
        OrderedDict("size" => 2^13, "sparsity" => 0.1),
        OrderedDict("size" => 2^20, "sparsity" => 3_000_000)
    ],
    "FEMLAB" => [
        "FEMLAB/poisson3Da",
        "FEMLAB/poisson3Db",
    ],
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
include("split_cols_finch_parallel_atomics.jl")
include("split_cols_finch_parallel_mutex.jl")
include("split_cols_static_scratchspace.jl")
include("split_nonzeros_static_scratchspace.jl")
include("split_cols_dynamic_grain_scratchspace.jl")
include("split_nonzeros_dynamic_grain_scratchspace.jl")
include("permute_split_rows_finch_parallel.jl")
include("permute_split_rows_dynamic_grain.jl")
include("spmv_taco.jl")

methods = OrderedDict(
    "serial_default_implementation" => serial_default_implementation_mul,
    "split_cols_finch_parallel_atomics" => split_cols_finch_parallel_atomics_mul,
    "split_cols_finch_parallel_mutex" => split_cols_finch_parallel_mutex_mul,
    "split_cols_static_scratchspace" => split_cols_static_scratchspace_mul,
    "split_cols_dynamic_grain_50_scratchspace" => split_cols_dynamic_grain_scratchspace_mul(50),
    "split_nonzeros_static_scratchspace" => split_nonzeros_static_scratchspace_mul,
    "split_nonzeros_dynamic_grain_500_scratchspace" => split_nonzeros_dynamic_grain_scratchspace_mul(500),
    "permute_split_rows_finch_parallel" => permute_split_rows_finch_parallel_mul,
    "permute_split_rows_dynamic_grain_50" => permute_split_rows_dynamic_grain_mul(50),
    "spmv_taco" => spmv_taco,
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
            A = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
        elseif dataset == "FEMLAB"
            A = matrixdepot(mtx)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        (num_rows, num_cols) = size(A)
        # x is a dense vector
        x = rand(num_cols)
        # y is the result vector
        y = zeros(num_rows)

        for (key, method) in methods
            result = method(y, A, x)

            if parsed_args["accuracy-check"]
                # Check the result of the multiplication
                serial_default_implementation_result = serial_default_implementation_mul(y, A, x)
                @assert norm(result.y - serial_default_implementation_result.y) / norm(serial_default_implementation_result.y) < 0.01 "Incorrect result for $key"
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
                write("results/spmv_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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



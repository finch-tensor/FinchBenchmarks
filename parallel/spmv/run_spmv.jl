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
# include("intrinsics_atomic_add.jl")
# include("atomix_atomic_add.jl")
include("separated_memory_add_static.jl")
include("separated_memory_add_balance_static.jl")
include("separated_memory_add_dynamic.jl")
# include("separate_sparselist_separated_memory_add_static.jl")

methods = OrderedDict(
    "serial_default_implementation" => serial_default_implementation_mul,
    # "intrinsics_atomic_add" => intrinsics_atomic_add_mul,
    # "atomix_atomic_add" => atomix_atomic_add_mul,
    "separated_memory_add_static" => separated_memory_add_static_mul,
    "separated_memory_add_balance_static" => separated_memory_add_balance_static_mul,
    "separated_memory_add_dynamic" => separated_memory_add_dynamic_mul,
    # "separate_sparselist_separated_memory_add_static" => separate_sparselist_separated_memory_add_static_mul,
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



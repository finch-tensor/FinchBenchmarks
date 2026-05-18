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
s = ArgParseSettings("Run Parallel Sum Experiments.")
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
    # set of same sizes, different sparsity
    "diff_sparsity" => [
        "Pajek/California", # 100m, 1e-5
        "Nasa/shuttle_eddy", # 100m, 1e-4
        "Nemeth/nemeth20", # 100m, 1e-3 (may have too many nnz)
        "Pothen/tandem_dual", # 10t, 1e-5
        "DIMACS10/G_n_pin_pout", # 10t, 1e-4
        "DNVS/m_t1", # 10t, 1e-3 
        "Mycielski/mycielskian17", # 10t, 1e-2 (may have too many nnz)
    ],
)

# Mapping from method keywords to methods
include("coalesce_impl.jl")
include("taco_impl.jl")
include("eigen_impl.jl")
include("mkl_impl.jl")


methods = OrderedDict(
    "coalesce_impl" => coalesce_impl,
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
        if dataset == "diff_sparsity"
            A = matrixdepot(mtx)
            v = vec(A)
            v = Tensor(SparseList(Element(0.0)), v)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(v, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the sum
                coalesce_impl_result = coalesce_impl(v, ncpu)

                rtol = 1e-6
                @assert isapprox(result.s, coalesce_impl_result.s, rtol=rtol) """
                    Incorrect result for $key: got $(result.s), expected $(coalesce_impl_result.s)
                    relative error = $(abs(result.s - coalesce_impl_result.s) / abs(coalesce_impl_result.s))
                    """
                # @assert result.s == coalesce_impl_result.s "Incorrect result for $key: $result : $coalesce_impl_result"
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



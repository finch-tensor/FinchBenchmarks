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
using SparseArrays

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
    "femlab" => [
	"FEMLAB/poisson3Db",
	"FEMLAB/sme3Dc",
	"FEMLAB/sme3Db",
	"FEMLAB/ns3Da",
	"FEMLAB/sme3Da",
	"FEMLAB/poisson3Da",
    ],
     "mirror" => [
        "SNAP/roadNet-CA",
        "SNAP/p2p-Gnutella31",
        "SNAP/cit-Patents",
        "SNAP/web-Google",
        "SNAP/amazon0312",
        "SNAP/wiki-Vote",
        "SNAP/email-Enron",
        "SNAP/ca-CondMat",
    ],
    "sparse" => [
        "SNAP/roadNet-CA",
        "SNAP/p2p-Gnutella31",
        "SNAP/cit-Patents",
        "SNAP/web-Google",
        "SNAP/amazon0312",
        "SNAP/wiki-Vote",
        "SNAP/email-Enron",
        "SNAP/ca-CondMat",
    ],
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
include("shard_implementation.jl")
include("eigen_impl.jl")
include("graphBLAS_impl.jl")

methods = OrderedDict(
    "shard_implementation" => shard_impl,
    "graphblas_impl" => graphblas_impl,
    "eigen_impl" => eigen_impl,
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
        if dataset == "mirror"
            A = SparseMatrixCSC(matrixdepot(mtx))
	        B = SparseMatrixCSC(matrixdepot(mtx))
        elseif dataset == "sparse"
            A = SparseMatrixCSC(matrixdepot(mtx))
	        B = A[randperm(size(A, 1)), randperm(size(A, 2))]
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
                ncpu = parsed_args["ncpu"]
                m, n = size(A)
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
                    write("results/hadamard_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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



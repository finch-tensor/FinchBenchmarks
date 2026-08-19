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
using LinearAlgebra

Random.seed!(1234)
MatrixDepot.update()

# Parsing Arguments
s = ArgParseSettings("Run Parallel SpMSpV Experiments.")
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
    # "uniform" => [
    #     # OrderedDict("size" => 1_000, "sparsity" => 0.01),
    #     OrderedDict("size" => 10_000, "sparsity" => 0.3),
    # ],
    # "hb_short" => [
    #     ("HB/bcsstm08", "bcsstm08"),
    #     ("HB/bcsstm09", "bcsstm09"),
    #     ("HB/bcsstm11", "bcsstm11"),
    #     ("HB/bcsstm26", "bcsstm26"),
    #     ("HB/bcsstm23", "bcsstm23"),
    #     ("HB/bcsstm25", "bcsstm25"),
    #     ("HB/bcsstk32", "bcsstk32"),
    #     ("HB/cegb2802", "cegb2802"),
    #     ("HB/bcsstk30", "bcsstk30"),
    #     ("HB/bcsstk31", "bcsstk31"),
    # ],
    "snap_large" => [
	    "SNAP/wiki-topcats",
        "SNAP/soc-Slashdot0811",
        "SNAP/p2p-Gnutella31",
        "SNAP/cit-Patents",
        "SNAP/web-Google",
        "SNAP/amazon0312",
        "SNAP/web-BerkStan",
        "SNAP/sx-stackoverflow",
        "SNAP/ca-CondMat",
    ],
    "snap_largest" => [
	    "SNAP/com-LiveJournal",
	    "SNAP/com-Orkut",
        "SNAP/soc-LiveJournal1",
        "SNAP/sx-stackoverflow",
        "SNAP/soc-Pokec",
        "SNAP/wiki-topcats",
        "SNAP/as-Skitter",
        "SNAP/cit-Patents",
    ],
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
include("coalesce_implementation.jl")
include("spmspv_eigen.jl")

methods = OrderedDict(
    "graphblas" => blas_spmspv,
    "coalesce_static" => coalesce_spmspv,
    "eigen" => spmspv_eigen,
    "coalesce_dynamic" => coalesce_spmspv_dynamic,
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
            x = fsprand(mtx["size"], mtx["sparsity"])
        elseif dataset == "snap_large" || dataset == "snap_largest"
	    @info "loading"
            A = SparseMatrixCSC(matrixdepot(mtx))
	    @info "A loaded"
            (m, n) = size(A)
            if m < 1000 || n < 1000
                continue
            end
            x = sprand(n, 0.1)
	    @info "x loaded"
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            result = method(A, x, Threads.nthreads())

            if parsed_args["accuracy-check"]
                # Check the result of the multiplication
                ref = serial_default_implementation_spmspv(A, x, 0)
                @info "checking accuracy"
                norm(SparseVector(ref.y) - SparseVector(result.y))/norm(SparseVector(ref.y)) < 0.01 || @warn("incorrect result via norm")
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
                write("results/spmspv_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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



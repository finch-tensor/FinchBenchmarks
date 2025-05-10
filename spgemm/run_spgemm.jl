#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    Pkg.status("Finch")
    println("Julia Version: $(VERSION)")
end

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using SparseArrays
using Finch
using LinearAlgebra
using Random

Random.seed!(1234)

s = ArgParseSettings("Run spgemm experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "spgemm_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "all"
end

parsed_args = parse_args(ARGS, s)

datasets = OrderedDict(
    "uniform" => [
        "uniform_dense",
        "uniform_sparse",
    ],
    "poisson" => [
        "FEMLAB/poisson3Da",
        "FEMLAB/poisson3Db",
    ],
)

include("spgemm_finch.jl")
include("spgemm_taco.jl")

methods = OrderedDict(
    "finch_custom_gustavson_dense" => spgemm_finch_custom_gustavson_dense,
    (has_taco() ? ["taco_gustavson" => spgemm_taco_gustavson] : [])...,
)

results = []

if parsed_args["dataset"] != "all"
    datasets = [(parsed_args["dataset"], datasets[parsed_args["dataset"]])]
end

for (dataset, mtxs) in datasets
    for mtx in mtxs
        if dataset == "uniform"
            if mtx == "uniform_dense"
                size = 10_000
                nnz = 1_000_000
                A = SparseMatrixCSC(fsprand(size, size, nnz))
                B = SparseMatrixCSC(fsprand(size, size, nnz))
            elseif mtx == "uniform_sparse"
                size = 10_000
                nnz = 30_000
                A = SparseMatrixCSC(fsprand(size, size, nnz))
                B = SparseMatrixCSC(fsprand(size, size, nnz))
            end
        else
            A = SparseMatrixCSC(matrixdepot(mtx))
            B = SparseMatrixCSC(matrixdepot(mtx))
        end

        C_ref = nothing
        for (key, method) in methods
            @info "testing" key mtx
            res = method(A, B)
            time = res.time
            C_ref = something(C_ref, res.C)

            norm(res.C - C_ref) / norm(C_ref) < 0.1 || @warn("incorrect result via norm")

            @info "results" time
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "kernel" => "spgemm",
                "matrix" => mtx,
                "dataset" => dataset,
                "num_threads" => Threads.nthreads(),
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end

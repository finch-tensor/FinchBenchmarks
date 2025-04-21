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
using Printf
using LinearAlgebra

s = ArgParseSettings("Run SPMV experiments.")

@add_arg_table! s begin
    "--output", "-o"
    arg_type = String
    help = "output file path"
    default = "spmv_results.json"
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

include("spmv_finch.jl")
include("spmv_taco.jl")

methods = OrderedDict(
    "finch_static_schedule" => spmv_finch_static,
    "finch_greedy_schedule" => spmv_finch_greedy,
    "finch_julia_schedule" => spmv_finch_julia,
    (has_taco() ? ["taco" => spmv_taco] : [])...,
)

results = []

int(val) = mod(floor(Int, val), Int8)

if parsed_args["dataset"] != "all"
    datasets = [(parsed_args["dataset"], datasets[parsed_args["dataset"]])]
end

for (dataset, mtxs) in datasets
    for mtx in mtxs
        if dataset == "uniform"
            if mtx == "uniform_dense"
                A = SparseMatrixCSC(fsprand(10_000, 10_000, 1_000_000))
            elseif mtx == "uniform_sparse"
                A = SparseMatrixCSC(fsprand(10_000, 10_000, 30_000))
            end
        else
            A = SparseMatrixCSC(matrixdepot(mtx))
        end

        (m, n) = size(A)
        x = rand(n)
        y = zeros(m)
        y_ref = nothing
        for (key, method) in methods
            @info "testing" key mtx
            res = method(y, A, x)
            time = res.time
            y_ref = something(y_ref, res.y)

            norm(res.y - y_ref) / norm(y_ref) < 0.1 || @warn("incorrect result via norm")

            @info "results" time
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "kernel" => "spmv",
                "matrix" => mtx,
                "dataset" => dataset,
                "num_threads" => Threads.nthreads(),
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end

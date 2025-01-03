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

#Here is where we use the julia arg parser to collect an input dataset keyword and an output file path

s = ArgParseSettings("Run spgemm experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "spgemm_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "zhang_small"
    "--batch", "-b"
        arg_type = Int
        help = "batch number"
        default = 1
    "--num_batches", "-B"
        arg_type = Int
        help = "number of batches"
        default = 1
    "--num_iters"
        arg_type = Int
        help = "number of iters to run"
        default = 20
    "--kernels"
        arg_type = String
        help = "set of kernels to run"
        default = "gustavson"
end

parsed_args = parse_args(ARGS, s)

num_iters = parsed_args["num_iters"]

datasets = Dict(
    "short" => [
        "HB/arc130",
    ],
    "scale" => [
        "file:./data/rand_128.ttx",
        "file:./data/rand_256.ttx",
        "file:./data/rand_512.ttx",
        "file:./data/rand_1024.ttx",
        "file:./data/rand_2048.ttx",
        "file:./data/rand_4096.ttx",
        "file:./data/rand_8192.ttx",
        "file:./data/rand_16384.ttx",
        "file:./data/rand_32768.ttx",
    ],
    "zhang_small" => [
        "SNAP/email-Eu-core",
        "SNAP/CollegeMsg",
        "SNAP/soc-sign-bitcoin-alpha",
        "SNAP/ca-GrQc",
        "SNAP/soc-sign-bitcoin-otc",
        "SNAP/p2p-Gnutella08",
        "SNAP/as-735",
        "SNAP/p2p-Gnutella09",
        "SNAP/wiki-Vote",
        "SNAP/p2p-Gnutella06",
        "SNAP/p2p-Gnutella05",
        "SNAP/ca-HepTh"
    ],
    "zhang_large" => [
        "FEMLAB/poisson3Da",
        "SNAP/wiki-Vote",
        "SNAP/email-Enron",
        "SNAP/ca-CondMat",
        "Oberwolfach/filter3D",
        "Williams/cop20k_A",
        "Um/offshore",
        "Um/2cubes_sphere",
        "vanHeukelum/cage12",
        "SNAP/amazon0312",
        "Hamm/scircuit",
        "SNAP/web-Google",
        "GHS_indef/mario002",
        "SNAP/cit-Patents",
        "JGD_Homology/m133-b3",
        "Williams/webbase-1M",
        "SNAP/roadNet-CA",
        "SNAP/p2p-Gnutella31",
        "Pajek/patents_main"
    ]
)

include("spgemm_finch.jl")
include("spgemm_taco.jl")
include("spgemm_eigen.jl")
include("spgemm_mkl.jl")

methods = Dict(
    "all" => [
        (has_taco() ? ["spgemm_taco_inner" => spgemm_taco_inner] : [])...,
        (has_taco() ? ["spgemm_taco_gustavson" => spgemm_taco_gustavson] : [])...,
        (has_taco() ? ["spgemm_taco_outer" => spgemm_taco_outer] : [])...,
        (has_eigen() ? ["spgemm_eigen" => spgemm_eigen] : [])...,
        (has_mkl() ? ["spgemm_mkl" => spgemm_mkl] : [])...,
        "spgemm_finch_inner" => spgemm_finch_inner,
        "spgemm_finch_gustavson" => spgemm_finch_gustavson,
        "spgemm_finch_outer" => spgemm_finch_outer,
        "spgemm_finch_outer_bytemap" => spgemm_finch_outer_bytemap,
        "spgemm_finch_outer_dense" => spgemm_finch_outer_dense,
    ],
    "fast" => [
        (has_taco() ? ["spgemm_taco_gustavson" => spgemm_taco_gustavson] : [])...,
        (has_eigen() ? ["spgemm_eigen" => spgemm_eigen] : [])...,
        (has_mkl() ? ["spgemm_mkl" => spgemm_mkl] : [])...,
        "spgemm_finch_gustavson" => spgemm_finch_gustavson,
    ],
)

results = []

batch = let 
    dataset = datasets[parsed_args["dataset"]]
    batch_num = parsed_args["batch"]
    num_batches = parsed_args["num_batches"]
    N = length(dataset)
    start_idx = min(fld1(N * (batch_num - 1) + 1, min(num_batches, N)), N + 1)
    end_idx = min(fld1(N * batch_num, min(num_batches, N)), N)
    dataset[start_idx:end_idx]
end

for mtx in batch
    if mtx[1:5] == "file:"
        A = SparseMatrixCSC(fread(mtx[6:end]))
    else
        A = SparseMatrixCSC(matrixdepot(mtx))
    end
    B = A
    C_ref = nothing
    for (key, method) in methods[parsed_args["kernels"]]
        @info "testing" key mtx
        res = method(A, B)
	C_ref = something(C_ref, SparseMatrixCSC(res.C))
	norm(C_ref - SparseMatrixCSC(res.C))/norm(C_ref) < 0.01 || @warn("incorrect result via norm")
        @info "results" res.time
        push!(results, OrderedDict(
            "time" => res.time,
            "method" => key,
            "kernel" => "spgemm",
            "matrix" => mtx,
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
end

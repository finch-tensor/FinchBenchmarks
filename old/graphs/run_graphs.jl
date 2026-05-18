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
using BenchmarkTools
using SparseArrays
using MatrixDepot
using Finch
using Graphs
using SimpleWeightedGraphs

s = ArgParseSettings("Run graph experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "graphs_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "willow"
    "--batch", "-b"
        arg_type = Int
        help = "batch number"
        default = 1
    "--num_batches", "-B"
        arg_type = Int
        help = "number of batches"
        default = 1
end

parsed_args = parse_args(ARGS, s)

include("datasets.jl")
include("bellmanford_finch.jl")
include("bfs_finch.jl")

include("bfs_lagraph.jl")
include("bellmanford_lagraph.jl")

function bfs_graphs(mtx)
    A = SimpleDiGraph(transpose(mtx))
    time = @belapsed Graphs.bfs_parents($A, 1)
    output = Graphs.bfs_parents(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function bellmanford_graphs(mtx)
    A = SimpleWeightedDiGraph(transpose(SparseMatrixCSC{Float64}(mtx)))
    time = @belapsed Graphs.bellman_ford_shortest_paths($A, 1)
    output = Graphs.bellman_ford_shortest_paths(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function check_bfs(A, src, res_parent, ref_parent)
    isnothing(res_parent) && return true # skip correctness check for LAGraph
    g = SimpleDiGraph(transpose(A))
    ref_levels = gdistances(g, src)
    for i in 1:nv(g)
        if ref_parent[i] == 0
            @assert res_parent[i] == 0
        elseif ref_parent[i] == i
            @assert res_parent[i] == i
        else
            @assert ref_levels[res_parent[i]] == ref_levels[i] - 1
        end
    end
    return true
end

function check_bellman(A, src, res, ref)
    isnothing(res) && return true # skip correctness check for LAGraph
    n = length(ref.dists)
    for i in 1:n
        if ref.dists[i] != res.dists[i]
            @info "dists" i res.dists[i] ref.dists[i]
        end
        if ref.parents[i] != 0
            @assert A[i, res.parents[i]] + ref.dists[res.parents[i]] == ref.dists[i]
        end
    end
    return true
end

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
    A = A + permutedims(A)
    (n, n) = size(A)
    for (op_name, check, methods) in [
        ("bfs",
            check_bfs,
            [
                "Graphs.jl" => bfs_graphs,
                "finch_push_pull" => bfs_finch_push_pull,
                "finch_push_only" => bfs_finch_push_only,
                "graphblas" => bfs_lagraph,
            ]
        ),
        ("bellmanford",
            check_bellman,
            [
                "Graphs.jl" => bellmanford_graphs,
                "Finch" => bellmanford_finch,
                "graphblas" => bellmanford_lagraph,
            ]
        ),
    ]
        if op_name == "bellmanford" && mtx in big_diameter
            continue
        end
        @info "testing" op_name mtx
        reference = nothing
        for (key, method) in methods
            result = method(A)

            time = result.time
            reference = something(reference, result.output)

            check(A, 1, result.output, reference) || @warn("incorrect result")

            # res.y == y_ref || @warn("incorrect result")
            @info "results" key result.time result.mem
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "operation" => op_name,
                "matrix" => mtx,
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end

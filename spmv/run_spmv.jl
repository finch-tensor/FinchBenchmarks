#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

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
    "willow_symmetric" => [
        "GHS_indef/exdata_1",
        #"Janna/Emilia_923",#too big
        #"Janna/Geo_1438",#too big
        "TAMU_SmartGridCenter/ACTIVSg70K"
    ],
    "willow_col_majmetric" => [
        "Goodwin/Goodwin_071", 
        #"Hamm/scircuit", #duplicate
        "LPnetlib/lpi_gran", #iffy
        "Norris/heart3",
        "Rajat/rajat26", 
        "TSOPF/TSOPF_RS_b678_c1" 
    ],
    "permutation" => [
        "permutation_synthetic"
    ], 
    "graph_symmetric" => [
        "SNAP/com-DBLP",
        "SNAP/email-Enron",
        "SNAP/ca-AstroPh",
    ],
    "graph_col_majmetric" => [
        "SNAP/soc-Epinions1",
    ],
    "banded" => [
        "toeplitz_small_band",
        "toeplitz_medium_band",
        "toeplitz_large_band",
    ],
    "triangle" => [
        "upper_triangle",
    ],
    "taco_symmetric" => [
        "HB/bcsstk17",
        "Williams/pdb1HYS",
        "Williams/cant",
        "Williams/consph",
        "Williams/cop20k_A",#iffy
        "DNVS/shipsec1",
        "Boeing/pwtk",#iffy
    ],
    "taco_col_majmetric" => [
        "Bova/rma10",
        "Williams/mac_econ_fwd500",
        "Williams/webbase-1M",#iffy
        "Hamm/scircuit",#iffy
    ],
)

include("synthetic.jl")
include("spmv_finch.jl")
include("spmv_taco.jl")
include("spmv_julia.jl")
include("spmv_eigen.jl")
include("spmv_mkl.jl")

dataset_tags = OrderedDict(
    "vuduc_symmetric" => "symmetric",
    "vuduc_col_majmetric" => "unsymmetric",
    "vuduc_symmetric_pattern" => "symmetric_pattern",
    "willow_symmetric" => "symmetric",
    "willow_col_majmetric" => "unsymmetric",
    "permutation" => "permutation",
    "banded" => "banded",
    "triangle" => "banded",
    "graph_symmetric" => "symmetric_pattern",
    "graph_col_majmetric" => "unsymmetric_pattern",
    "taco_symmetric" => "symmetric",
    "taco_col_majmetric" => "unsymmetric",
)

methods = OrderedDict(
    "symmetric" => [
        #"julia_stdlib" => spmv_julia,
        #"finch_sym_sparselist" => spmv_finch_sym_sparselist,
        #"finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        #"finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        #"finch_sym_sparseblocklist" => spmv_finch_sym_sparseblocklist,
        #"finch_col_maj_sparseblocklist" => spmv_finch_col_maj_sparseblocklist,
        #"finch_row_maj_sparseblocklist" => spmv_finch_row_maj_sparseblocklist,
        #"taco_col_maj" => spmv_taco_col_maj,
        #"taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
    "unsymmetric" => [
        "julia_stdlib" => spmv_julia,
        "finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        "finch_col_maj_sparseblocklist" => spmv_finch_col_maj_sparseblocklist,
        "finch_row_maj_sparseblocklist" => spmv_finch_row_maj_sparseblocklist,
        "taco_col_maj" => spmv_taco_col_maj,
        "taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
    "symmetric_pattern" => [
        "julia_stdlib" => spmv_julia,
        "finch_sym_sparselist" => spmv_finch_sym_sparselist,
        "finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        "finch_sym_sparselist_pattern" => spmv_finch_sym_sparselist_pattern,
        "finch_col_maj_sparselist_pattern" => spmv_finch_col_maj_sparselist_pattern,
        "finch_row_maj_sparselist_pattern" => spmv_finch_row_maj_sparselist_pattern,
        "finch_sym_sparseblocklist" => spmv_finch_sym_sparseblocklist,
        "finch_col_maj_sparseblocklist" => spmv_finch_col_maj_sparseblocklist,
        "finch_row_maj_sparseblocklist" => spmv_finch_row_maj_sparseblocklist,
        "taco_col_maj" => spmv_taco_col_maj,
        "taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
    "unsymmetric_pattern" => [
        "julia_stdlib" => spmv_julia,
        "finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        "finch_col_maj_sparselist_pattern" => spmv_finch_col_maj_sparselist_pattern,
        "finch_row_maj_sparselist_pattern" => spmv_finch_row_maj_sparselist_pattern,
        "finch_col_maj_sparseblocklist" => spmv_finch_col_maj_sparseblocklist,
        "finch_row_maj_sparseblocklist" => spmv_finch_row_maj_sparseblocklist,
        "taco_col_maj" => spmv_taco_col_maj,
        "taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
    "permutation" => [
        "julia_stdlib" => spmv_julia,
        "finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        "finch_col_maj_sparselist_pattern" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist_pattern" => spmv_finch_row_maj_sparselist,
        "finch_col_maj_sparsepoint_pattern" => spmv_finch_col_maj_sparsepoint_pattern,
        "finch_row_maj_sparsepoint_pattern" => spmv_finch_row_maj_sparsepoint_pattern,
        "taco_col_maj" => spmv_taco_col_maj,
        "taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
    "banded" => [
        "julia_stdlib" => spmv_julia,
        "finch_col_maj_sparselist" => spmv_finch_col_maj_sparselist,
        "finch_row_maj_sparselist" => spmv_finch_row_maj_sparselist,
        "finch_col_maj_sparseband" => spmv_finch_col_maj_sparseband,
        "finch_row_maj_sparseband" => spmv_finch_row_maj_sparseband,
        "taco_col_maj" => spmv_taco_col_maj,
        "taco_row_maj" => spmv_taco_row_maj,
        "eigen" => spmv_eigen,
        "mkl" => spmv_mkl,
    ],
)

results = []

int(val) = mod(floor(Int, val), Int8)

if parsed_args["dataset"] != "all"
	datasets = [(parsed_args["dataset"], datasets[parsed_args["dataset"]])]
end

for (dataset, mtxs) in datasets
    tag = dataset_tags[dataset]
    for mtx in mtxs
        if dataset == "permutation"
            A = SparseMatrixCSC(reverse_permutation_matrix(200000))
        elseif dataset == "banded"
            if mtx == "toeplitz_small_band"
                A = SparseMatrixCSC(banded_matrix(10000, 5))
            elseif mtx == "toeplitz_medium_band"
                A = SparseMatrixCSC(banded_matrix(10000, 30))
            elseif mtx == "toeplitz_large_band"
                A = SparseMatrixCSC(banded_matrix(10000, 100))
            end
        elseif dataset == "triangle"
            if mtx == "upper_triangle"
                A = SparseMatrixCSC(upper_triangle_matrix(1024))
            end
        else
            A = SparseMatrixCSC(matrixdepot(mtx))
        end

        # A = map((val) -> int(val), A)
        (m, n) = size(A)
        x = rand(n)
        # x = sprand(n, 0.1)
        y = zeros(m)
        y_ref = nothing
        for (key, method) in methods[tag]
            @info "testing" key mtx
            res = method(y, A, x)
            #=
                rm(key * "_results.txt", force=true)
                open(key * "_results.txt","a") do io
                    for i = 1:n
                        @printf(io,"%f\n", res.y[i])
                    end
                end
            =#
            time = res.time
            y_ref = something(y_ref, res.y)

            norm(res.y - y_ref)/norm(y_ref) < 0.1 || @warn("incorrect result via norm")

            # res.y == y_ref || @warn("incorrect result")
            @info "results" time
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "kernel" => "spmv",
                "matrix" => mtx,
                "dataset" => dataset,
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end

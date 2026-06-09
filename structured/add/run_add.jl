#!/usr/bin/env julia
# using Base: nothing_sentinel
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using FileIO
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Images
using Random

Random.seed!(1234)

# Parsing Arguments
s = ArgParseSettings("Run Structured Add Experiments.")
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
    "image" => [
        "very_highly_compressed/www.abalip.com.jpg",
        "very_highly_compressed/www.carmelmusic.com.jpg",
        "very_highly_compressed/www.claudiozappi.it.jpg",
        "very_highly_compressed/www.duo-thais.com.jpg", 
        "very_highly_compressed/www.handball-riehen.ch.jpg",
        # "../old/images_morphology/masks/Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).png",
        "../old/images_morphology/masks/Fig0227(a)(washington_infrared).png",
        "../old/images_morphology/masks/Fig1001(b)(edge_image).png",
        "../old/images_morphology/masks/Fig1213(e)(Mask_B1_without_numbers).png",
        "../old/images_morphology/masks/FigP0311.png",
    ],
    # "uniform" => [
    #     OrderedDict("size" => 50_000, "sparsity" => 0.0001),
    # ],
)

include("shard_impl.jl")
include("cv_impl.jl")


methods = OrderedDict(
    "shard_impl" => shard_impl,
    "cv_impl" => add_cv_impl,
)

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for item in mtxs
        matrix_label = ""

        if dataset == "image"
            a_path = item
            A_orig = load("../" * a_path)
            m, n = size(A_orig)
            A = zeros(m, n)

            for i in 1:m
                for j in 1:n
                    c = A_orig[i, j]
                    # Handle both RGB images and grayscale images
                    try
                        A[i, j] = c.r * 0.299 + c.g * 0.587 + c.b * 0.114
                    catch
                        # Grayscale or other single-channel format
                        A[i, j] = float(c)
                    end
                end
            end

            B = reverse(A, dims=2)

            matrix_label = split(a_path, "/")[end]
            
        elseif dataset == "uniform"
            mtx = item
            A = fsprand(Float64, mtx["size"], mtx["size"], mtx["sparsity"])
            B = fsprand(Float64, mtx["size"], mtx["size"], mtx["sparsity"])
            matrix_label = string("uniform size=", mtx["size"], " sparsity=", mtx["sparsity"])
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            result = method(A, B, ncpu)

            if parsed_args["accuracy-check"]
                # Check the result of the add
                shard_impl_result = shard_impl(A, B, ncpu)

                m, n = size(shard_impl_result.C)
                expected = shard_impl_result.C
                for i in 1:m
                    for j in 1:n
                        @assert isapprox(result.C[i,j], expected[i,j]) "Incorrect result for $key at ($i,$j): got $(result.C[i,j]), expected $(expected[i,j])"
                    end
                end
            end

            # Write result
            time = result.time
            @info "result for $key on $matrix_label" time
            push!(results, OrderedDict(
                "time" => time,
                "n_threads" => Threads.nthreads(),
                "method" => key,
                "dataset" => dataset,
                "matrix" => matrix_label,
            ))

            if isnothing(parsed_args["output"])
                write("results/add_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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

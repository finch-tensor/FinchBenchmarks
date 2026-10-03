#!/usr/bin/env julia
# using Base: nothing_sentinel
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
    action = :append_arg
    help = "method keyword (repeat -m to run several; default: all)"
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
    ],
    # "uniform" => [
    #     OrderedDict("size" => 1_000, "sparsity" => 0.1),
    #     OrderedDict("size" => 1_000, "sparsity" => 0.0001),
    #     OrderedDict("size" => 100_000, "sparsity" => 0.00001),
    # ],
)

include("coalesce_impl.jl")
include("wingspan_hist.jl")
include("birdseed_hist.jl")
include("halide_impl.jl")

methods = OrderedDict(
    "coalesce_impl" => coalesce_impl,
    "wingspan_hist" => wingspan_hist,
    "birdseed_hist" => birdseed_hist,
    "coalesce_impl_gray" => coalesce_impl_gray,
    "wingspan_hist_gray" => wingspan_hist_gray,
    "birdseed_hist_gray" => birdseed_hist_gray,
    "halide_hist" => hist_halide_impl,
)

selected = something(parsed_args["method"], String[])
if !isempty(selected)
    for method_name in selected
        @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    end
    methods = OrderedDict(
        method_name => methods[method_name] for method_name in selected
    )
end

# Ground truth: count of each raw pixel value (0-based), keyed by Int or (r, g, b)
function hist_counts(input)
    counts = Dict{Any,Int}()
    for px in input
        k = px isa Tuple ? Int.(px) : Int(px)
        counts[k] = get(counts, k, 0) + 1
    end
    return counts
end

function check_hist(key, hist, expected, npixels)
    @assert sum(values(expected)) == npixels
    if startswith(key, "halide")
        nz = ffindnz(hist)
        got = Dict{Any,Int}()
        for p in eachindex(nz[end])
            c = ntuple(d -> Int(nz[d][p]) - 1, length(nz) - 1)
            got[length(c) == 1 ? c[1] : c] = Int(nz[end][p])
        end
        for (k, v) in expected
            @assert get(got, k, 0) == v "Incorrect result for $key at $k: got $(get(got, k, 0)), expected $v"
        end
        println("correct 1")
        @assert sum(values(got)) == npixels "Incorrect total for $key: got $(sum(values(got))), expected $npixels"
        println("correct 2")
    else
        for (k, v) in expected
            if k isa Tuple && (startswith(key, "birdseed") || startswith(key, "wingspan"))
                # (r, g, b) packed into one 1-based key: (r << 16) | (g << 8) | b
                got = hist[((k[1] << 16) | (k[2] << 8) | k[3]) + 1]
            elseif startswith(key, "birdseed")
                got = hist[k + 1]
            else
                got = hist[k...]
            end
            @assert got == v "Incorrect result for $key at $k: got $got, expected $v"
        end
    end
end

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        # Get relevant matrix
        if dataset == "image"
            A_orig = load("../" * mtx)
            m, n = size(A_orig)
            A = fill((UInt8(0), UInt8(0), UInt8(0)), m, n)
            A_gray = fill(UInt8(0), m, n)

            for i in 1:m
                for j in 1:n
                    c = A_orig[i, j]

                    A[i,j] = (UInt8(c.r * 255), UInt8(c.g * 255), UInt8(c.b * 255))
                    try
                        A_gray[i, j] = round(UInt8, (c.r * 0.299 + c.g * 0.587 + c.b * 0.114) * 255)
                    catch
                        # Grayscale or other single-channel format
                        A_gray[i, j] = round(UInt8, float(c) * 255)
                    end
                end
            end
        elseif dataset == "uniform"
            R = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            G = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            B = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])
            A = fill((UInt8(0), UInt8(0), UInt8(0)), mtx["size"], mtx["size"])
            for i in 1:mtx["size"]
                for j in 1:mtx["size"]
                    A[i, j] = (R[i, j], G[i, j], B[i, j])
                end
            end
            A_gray = fsprand(UInt8, mtx["size"], mtx["size"], mtx["sparsity"])

        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            ncpu = parsed_args["ncpu"]
            gray = endswith(key, "_gray")
            input = gray ? A_gray : A
            result = method(input, ncpu)

            if parsed_args["accuracy-check"]
                check_hist(key, result.hist, hist_counts(input), length(input))
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
                write("results/hist_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
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

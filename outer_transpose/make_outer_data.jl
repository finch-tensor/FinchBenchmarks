#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(dirname(@__DIR__), "envs", "birdseed"))
    Pkg.instantiate()
    Pkg.status("Finch")
    println("Julia Version: $(VERSION)")
end

using Finch
using TensorMarket
using SparseArrays, Random

const M       = 4194304
const K       = 131072
const NACTIVE = 65536      # active columns in each of A and B
const NSHARED = 64         # columns active in both A and B
const NNZCOL  = 16         # nonzeroes per active column

@assert NSHARED <= NACTIVE "more shared columns than active columns"
@assert 2 * NACTIVE - NSHARED <= K "not enough columns for $NSHARED shared + disjoint remainders"

function randcols(rng, cols)
    cols = sort(cols)
    colptr = zeros(Int, K + 1)
    rowval = Int[]
    sizehint!(rowval, NNZCOL * length(cols))
    for j in cols
        colptr[j+1] = NNZCOL
        rows = Int[]
        while length(rows) < NNZCOL
            i = rand(rng, 1:M)
            i in rows || push!(rows, i)
        end
        append!(rowval, sort!(rows))
    end
    colptr[1] = 1
    cumsum!(colptr, colptr)
    return SparseMatrixCSC(M, K, colptr, rowval, rand(rng, length(rowval)))
end

rng = MersenneTwister(42)
p = randperm(rng, K)
shared = p[1:NSHARED]
only_a = p[NSHARED+1:NACTIVE]
only_b = p[NACTIVE+1:2*NACTIVE-NSHARED]
A = randcols(rng, vcat(shared, only_a))
B = randcols(rng, vcat(shared, only_b))
@assert length(intersect(findall(>(0), diff(A.colptr)), findall(>(0), diff(B.colptr)))) == NSHARED

datadir = joinpath(@__DIR__, "data")
mkpath(datadir)
for (name, X) in (("A", A), ("B", B))
    fn = joinpath(datadir, "outer_$(name).ttx")
    fwrite(fn, Tensor(Dense(SparseList(Element(0.0))), X))   # Dense(SparseList) = column-major (CSC)
    println("$fn  $(M)x$(K) nnz=$(nnz(X)) active_cols=$(count(>(0), diff(X.colptr)))")
end
println("shared active k = $NSHARED, flops = $(NSHARED * NNZCOL * NNZCOL)")

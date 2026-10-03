#!/usr/bin/env julia
# C_ij = sum_k A_ik B_jk, A and B both M x K, column-major (CSC). Each k is one outer product A(:,k) * B(:,k)'.
# A and B each have NACTIVE nonempty columns, exactly NSHARED of them common to both (so only NSHARED
# k iterations do any work); every active column holds NNZCOL nonzeroes at distinct uniform random rows.
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    Pkg.status("Finch")
    println("Julia Version: $(VERSION)")
end

using Finch
using TensorMarket
using SparseArrays, Random

const M       = 4096
const K       = 131072
const NACTIVE = 65536       # active columns per matrix
const NSHARED = 64          # columns active in both A and B
const NNZCOL  = 32          # nonzeroes per active column

@assert 2NACTIVE - NSHARED <= K "not enough columns for $NSHARED shared + disjoint remainders"

function randcols(rng, cols)
    I = Int[]; J = Int[]
    sizehint!(I, NNZCOL * length(cols)); sizehint!(J, NNZCOL * length(cols))
    for j in cols
        rows = Set{Int}()
        while length(rows) < NNZCOL; push!(rows, rand(rng, 1:M)); end
        append!(I, rows); append!(J, fill(j, NNZCOL))
    end
    return sparse(I, J, rand(rng, length(I)), M, K)
end

rng = MersenneTwister(42)
p = randperm(rng, K)
shared = p[1:NSHARED]
only_a = p[NSHARED+1:NACTIVE]
only_b = p[NACTIVE+1:2NACTIVE-NSHARED]
A = randcols(rng, vcat(shared, only_a))
B = randcols(rng, vcat(shared, only_b))
@assert length(intersect(findall(>(0), diff(A.colptr)), findall(>(0), diff(B.colptr)))) == NSHARED

mkpath("data")
for (name, X) in (("A", A), ("B", B))
    fn = "data/outer_$(name).ttx"
    fwrite(fn, Tensor(Dense(SparseList(Element(0.0))), X))   # Dense(SparseList) = column-major (CSC)
    println("$fn  $(M)x$(K) nnz=$(nnz(X)) active_cols=$(count(>(0), diff(X.colptr)))")
end
println("shared active k = $NSHARED, flops = $(NSHARED * NNZCOL^2)")

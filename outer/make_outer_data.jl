#!/usr/bin/env julia
# C_ij = sum_k A_ik B_jk, A and B both M x K, column-major (CSC). Each k is one outer product A(:,k) * B(:,k)'.
# A has NACTIVE_A nonempty columns and B has NACTIVE_B, exactly NSHARED of them common to both (so only
# NSHARED k iterations do any work); every active column of A (B) holds NNZCOL_A (NNZCOL_B) nonzeroes at
# distinct uniform random rows. A is heavy and B touches only the shared columns, so Gustavson (MKL) must
# stream all of A while the outer product only visits the shared k.
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

const M         = 16384
const K         = 131072
const NACTIVE_A = 131072     # active columns in A
const NACTIVE_B = 4          # active columns in B
const NSHARED   = 4          # columns active in both A and B
const NNZCOL_A  = 256        # nonzeroes per active column of A
const NNZCOL_B  = 32         # nonzeroes per active column of B

@assert NSHARED <= min(NACTIVE_A, NACTIVE_B) "more shared columns than active columns"
@assert NACTIVE_A + NACTIVE_B - NSHARED <= K "not enough columns for $NSHARED shared + disjoint remainders"

function randcols(rng, cols, nnzcol)
    I = Int[]; J = Int[]
    sizehint!(I, nnzcol * length(cols)); sizehint!(J, nnzcol * length(cols))
    for j in cols
        append!(I, randperm(rng, M)[1:nnzcol]); append!(J, fill(j, nnzcol))
    end
    return sparse(I, J, rand(rng, length(I)), M, K)
end

rng = MersenneTwister(42)
p = randperm(rng, K)
shared = p[1:NSHARED]
only_a = p[NSHARED+1:NACTIVE_A]
only_b = p[NACTIVE_A+1:NACTIVE_A+NACTIVE_B-NSHARED]
A = randcols(rng, vcat(shared, only_a), NNZCOL_A)
B = randcols(rng, vcat(shared, only_b), NNZCOL_B)
@assert length(intersect(findall(>(0), diff(A.colptr)), findall(>(0), diff(B.colptr)))) == NSHARED

datadir = joinpath(@__DIR__, "data")
mkpath(datadir)
for (name, X) in (("A", A), ("B", B))
    fn = joinpath(datadir, "outer_$(name).ttx")
    fwrite(fn, Tensor(Dense(SparseList(Element(0.0))), X))   # Dense(SparseList) = column-major (CSC)
    println("$fn  $(M)x$(K) nnz=$(nnz(X)) active_cols=$(count(>(0), diff(X.colptr)))")
end
println("shared active k = $NSHARED, flops = $(NSHARED * NNZCOL_A * NNZCOL_B)")

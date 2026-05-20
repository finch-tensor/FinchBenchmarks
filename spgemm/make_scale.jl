#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    Pkg.status("Finch")
    println("Julia Version: $(VERSION)")
end

using Finch
using TensorMarket

mkpath("data")

for N = 7:14
    m = n = 2^N
    nnz = 4*m
    fwrite("data/rand_$(m).ttx", fsprand(m, n, nnz))
end



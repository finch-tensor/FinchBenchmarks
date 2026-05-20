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

mkpath("hypersparse")

for N = 7:14
    m = n = 2^N
    nnz = m / 2
    fwrite("hypersparse/rand_$(m).ttx", fsprand(m, n, nnz))
end



using SuiteSparseGraphBLAS
using SparseArrays
using BenchmarkTools

function graphblas_impl(A, B, num_cpu)
    SuiteSparseGraphBLAS.gbset(:nthreads, num_cpu)

    _A = GBMatrix{Float64}(Float64.(A))
    _B = GBMatrix{Float64}(Float64.(B))
    
    _C = _A * _B

    time = @belapsed $_A * $_B

    C_sparse = SparseMatrixCSC(_C)
    return (; time=time, C=C_sparse)
end

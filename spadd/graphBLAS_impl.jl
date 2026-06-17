using SuiteSparseGraphBLAS
using SparseArrays
using BenchmarkTools

function graphblas_impl(A, B, num_cpu)
    SuiteSparseGraphBLAS.gbset(:nthreads, num_cpu)

    _A = GBMatrix(A)
    _B = GBMatrix(B)
    
    _C = _A .+ _B

    time = @belapsed $_A .+ $_B

    C_sparse = SparseMatrixCSC(_C)
    return (; time=time, C=C_sparse)
end
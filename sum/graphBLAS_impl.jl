using SuiteSparseGraphBLAS
using SparseArrays
using BenchmarkTools

function graphblas_impl(v1, v2, num_cpu)
    SuiteSparseGraphBLAS.gbset(:nthreads, num_cpu)

    spv = SparseVector(v1)
    idx, val = findnz(spv)

    gv1 = GBVector(idx, val, length(spv))

    spv = SparseVector(v2)
    idx, val = findnz(spv)

    gv2 = GBVector(idx, val, length(spv))

    acc = reduce(+, gv1 .* gv2)

    time = @belapsed reduce(+, $gv1 .* $gv2)

    return (; time=time, s=acc)
end
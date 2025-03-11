using Finch
using BenchmarkTools

include("utils/merge.jl")
include("utils/permutation.jl")

function graph_partition_weighted_reorder_merge(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)

    perm = create_weighted_permutation(_A)
    _A = matrix_col_permutation(_A, perm)
    time = @belapsed merge_helper($_y, $_A, $_x)
    _y = vector_permutation(_y, invperm(perm))
    return (; time=time, y=_y)
end


using Finch
using BenchmarkTools

include("utils/merge.jl")

function merge(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)

    time = @belapsed merge_swizzle_helper($_y, $_A, $_x)
    return (; time=time, y=_y)
end


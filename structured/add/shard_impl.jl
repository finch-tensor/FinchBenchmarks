using Finch
using BenchmarkTools
using SparseArrays


function shard_impl(A, B, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element(0.0))), A)
    _B = Tensor(Dense(SparseRunList(Element(0.0))), B)
    _C = Tensor(Dense(Shard(dev, SparseRunList(Element(0.0)))))

    # println("Before: A[244,685] = " * string(_A[244,685]))
    # println("Before: B[244,685] = " * string(_B[244,685]))
    # println("Before: C[244,685] = " * string(_C[244,685]))

    time = @belapsed begin
        (_A, _B, _C, dev) = $(_A, _B, _C, dev)
        @finch mode = :fast begin
            _C .= 0.0
            for j = parallel(_, dev), i = _
                _C[i, j] = _A[i, j] + _B[i, j]
            end
        end
    end

    @finch mode = :fast begin
        _C .= 0.0
        for j = parallel(_, dev), i = _
            _C[i, j] = _A[i, j] + _B[i, j]
        end
    end

    # println("Final: A[244,685] = " * string(_A[244,685]))
    # println("Final: B[244,685] = " * string(_B[244,685]))
    # println("Final: C[244,685] = " * string(_C[244,685]))

    return (; time=time, C = _C)
end

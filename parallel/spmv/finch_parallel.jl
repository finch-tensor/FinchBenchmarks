using Finch
using BenchmarkTools


function finch_parallel(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                @finch mode = :fast begin
                        _y .= 0
                        for i = parallel(_), j = _
                                _y[i] += _A[i, j] * _x[j]
                        end
                end
        end
        return (; time=time, y=_y)
end

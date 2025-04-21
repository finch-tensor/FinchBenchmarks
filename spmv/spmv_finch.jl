using Finch
using BenchmarkTools

function spmv_finch_static(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    time = @belapsed begin
        (_y, _A, _x) = $(_y, _A, _x)
        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu(), static_schedule()), i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end
    return (; time=time, y=_y)
end

function spmv_finch_greedy(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    time = @belapsed begin
        (_y, _A, _x) = $(_y, _A, _x)
        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu(), greedy_schedule(100)), i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end
    return (; time=time, y=_y)
end

function spmv_finch_julia(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    time = @belapsed begin
        (_y, _A, _x) = $(_y, _A, _x)
        @finch mode = :fast begin
            _y .= 0
            for j = parallel(_, cpu(), julia_schedule(100)), i = _
                _y[j] += _A[i, j] * _x[i]
            end
        end
    end
    return (; time=time, y=_y)
end

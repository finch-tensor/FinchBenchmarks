using Finch
using BenchmarkTools

function spgemm_finch_static_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    _C = Tensor(Dense(Dense(Element(0.0))), zeros(size(A)[1], size(B)[2]))
    _w = Tensor(SparseByteMap(Element(0.0)))
    time = @belapsed begin
        (_A, _B, _C, _w) = $(_A, _B, _C, _w)
        @finch mode = :fast begin
            _C .= 0
            for j in parallel(_, cpu(), static_schedule())
                _w .= 0
                for k in _, i in _
                    _w[i] += _A[i, k] * _B[k, j]
                end
                for i in _
                    _C[i, j] = _w[i]
                end
            end
        end
    end
    # return (; time=time, C=_C)
    return (; time=time,)
end

function spgemm_finch_greedy_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    _C = Tensor(Dense(Dense(Element(0.0))), zeros(size(A)[1], size(B)[2]))
    _w = Tensor(SparseByteMap(Element(0.0)))
    time = @belapsed begin
        (_A, _B, _C, _w) = $(_A, _B, _C, _w)
        @finch mode = :fast begin
            _C .= 0
            for j in parallel(_, cpu(), greedy_schedule(100))
                _w .= 0
                for k in _, i in _
                    _w[i] += _A[i, k] * _B[k, j]
                end
                for i in _
                    _C[i, j] = _w[i]
                end
            end
        end
    end
    # return (; time=time, C=_C)
    return (; time=time,)
end

using Finch
using BenchmarkTools


function serial_default_implementation_spmspv(A, x, nt)
        _y = Tensor(SparseDict(Element(0.0)))
        _x = Tensor(SparseList(Element(0.0)), x)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        time = @belapsed begin
                (_A, _x, _y) = $(_A, _x, _y)
                @finch mode = :fast begin
                        _y .= 0
                        for j = _, i = _
                                _y[i] += _A[i, j] * _x[j]
                        end
                end
        end

        @finch mode = :fast begin
                _y .= 0
                for j = _, i = _
                        _y[i] += _A[i, j] * _x[j]
                end
        end
        return (; time=time, y=_y)
end

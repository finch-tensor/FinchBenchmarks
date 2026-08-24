using Finch
using BenchmarkTools
using SparseArrays


function finch_impl(B, C, D, num_cpu)
    dev = cpu(:t, num_cpu)
    _B = Tensor(SparseList(SparseList(SparseList(Element(0.0)))), B)
    _C = Tensor(Dense(SparseList(Element(0.0))), C)
    _D = Tensor(Dense(SparseList(Element(0.0))), D)
    
    _A = Tensor(Dense(Shard(dev, SparseByteMap(Element(0.0)))))
    time = @belapsed begin
        (_A, _B, _C, _D, dev) = $(_A, _B, _C, _D, dev)
        @finch mode = :fast begin
            _A .= 0
           for r = parallel(_, dev)
                for k = _
                    for j = _
                        for i = _
                            A[i, r] += _B[i, j, k] * _D[k, r] * _C[j, r]
                        end
                    end
                end
            end
        end
    end

   _A = Tensor(Dense(Shard(dev, SparseByteMap(Element(0.0)))))

    @finch begin
        _A .= 0
        for r = parallel(_, dev)
            for k = _
                for j = _
                    for i = _
                        w[i] += _B[i, j, k] * _D[k, r] * _C[j, r]
                    end
                end
            end
        end
    end

    return (; time=time, A=_A)
end

using Finch
using BenchmarkTools
using Base.Threads


function shard_add(A, B, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    cpu_dev = cpu(:id, num_cpu)
    _C = Tensor(Dense(Shard(cpu_dev, SparseList(Element(0.0)))))

    time = @belapsed begin
        (_A, _B, _C, cpu_dev) = $(_A, _B, _C, cpu_dev)

        @finch mode = :fast begin
            _C .= 0
            for j = parallel(_, cpu_dev)
                for i = _
                    _C[i, j] = _A[i, j] + _B[i, j]
                end
            end
        end
    end

    @finch mode = :fast begin
        _C .= 0
        for j = parallel(_, cpu_dev)
            for i = _
                _C[i, j] = _A[i, j] + _B[i, j]
            end
        end
    end

    return (; time=time, C=_C)
end

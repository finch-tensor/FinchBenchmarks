using Finch
using BenchmarkTools
using Base.Threads

# Int32 indices / Float32 values, matching nacho's i32/f32 kernel.
birdseed_fmt() = SparseList{Int32}(SparseList{Int32}(Element{0.0f0,Float32,Int32}()))

function birdseed_spadd(A, B, num_cpu)
    _A = Tensor(birdseed_fmt(), A)
    _B = Tensor(birdseed_fmt(), B)

    cpu_dev = cpu(:id, num_cpu)
    _C = Tensor(Coalesce(cpu_dev, birdseed_fmt(); mode=:fast))
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

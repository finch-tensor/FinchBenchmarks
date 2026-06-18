using Finch
using BenchmarkTools
using Base.Threads


function shard_impl(A, B, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    cpu_dev = cpu(:id, num_cpu)
    _C = Tensor(Dense(Shard(cpu_dev, SparseList(Element(0.0)))))
    chk = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    opt = 99999999
    for size in chk
        sch = greedy_schedule(size)
        @info "testing $size"
        time = @belapsed begin
            (_A, _B, _C, cpu_dev, sch) = $(_A, _B, _C, cpu_dev, sch)

            @finch mode = :fast begin
                _C .= 0
                for j = parallel(_, cpu_dev, sch)
                    for i = _
                        _C[i, j] = _A[i, j] * _B[i, j]
                    end
                end
            end
        end
        @info "time: $time"
        if time < opt
            opt = time
        end
    end
    @info "testing static"
    time = @belapsed begin
        (_A, _B, _C, cpu_dev) = $(_A, _B, _C, cpu_dev)

        @finch mode = :fast begin
            _C .= 0
            for j = parallel(_, cpu_dev)
                for i = _
                    _C[i, j] = _A[i, j] * _B[i, j]
                end
            end
        end
    end
    
    if time < opt
        opt = time
    end

    @finch mode = :fast begin
        _C .= 0
        for j = parallel(_, cpu_dev)
            for i = _
                _C[i, j] = _A[i, j] * _B[i, j]
            end
        end
    end

    return (; time=opt, C=_C)
end


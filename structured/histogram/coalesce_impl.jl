using Finch
using BenchmarkTools
using SparseArrays


function coalesce_impl(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element((UInt8(0), UInt8(0), UInt8(0))))), A)
    _hist = Tensor(Coalesce(dev, SparseDict(SparseDict(SparseDict(Element(0))))))

    time = @belapsed begin
        (_A, _hist, dev) = $(_A, _hist, dev)
        @finch mode = :fast begin
            _hist .= 0
            for j = parallel(_, dev), i = _
                let r = getindex(_A[i, j], 1), g = getindex(_A[i, j], 2), b = getindex(_A[i, j], 3)
                    _hist[r, g, b] += 1
                end
            end
        end
    end

    @finch mode = :fast begin
        _hist .= 0
        for j = parallel(_, dev), i = _
            let r = getindex(_A[i, j], 1), g = getindex(_A[i, j], 2), b = getindex(_A[i, j], 3)
                _hist[r, g, b] += 1
            end
        end
    end

    return (; time=time, hist = _hist)
end

function coalesce_impl_ddd(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element((UInt8(0), UInt8(0), UInt8(0))))), A)
    _hist = Tensor(Coalesce(dev, Dense(Dense(Dense(Element(0), 256), 256), 256)))

    time = @belapsed begin
        (_A, _hist, dev) = $(_A, _hist, dev)
        @finch mode = :fast begin
            _hist .= 0
            for j = parallel(_, dev), i = _
                let r = getindex(_A[i, j], 1), g = getindex(_A[i, j], 2), b = getindex(_A[i, j], 3)
                    _hist[r, g, b] += 1
                end
            end
        end
    end

    @finch mode = :fast begin
        _hist .= 0
        for j = parallel(_, dev), i = _
            let r = getindex(_A[i, j], 1), g = getindex(_A[i, j], 2), b = getindex(_A[i, j], 3)
                _hist[r, g, b] += 1
            end
        end
    end

    return (; time=time, hist = _hist)
end

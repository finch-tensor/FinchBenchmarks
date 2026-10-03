using Finch
using BenchmarkTools
using SparseArrays


function birdseed_hist(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element((UInt8(0), UInt8(0), UInt8(0))))), A)
    # (r, g, b) is packed into a single Int64 key: (r << 16) | (g << 8) | b, 1-based.
    _hist = Tensor(Coalesce(dev, SparseHash(Element(0), 1 << 24)))

    time = @belapsed begin
        (_A, _hist, dev) = $(_A, _hist, dev)
        @finch mode = :fast begin
            _hist .= 0
            for j = parallel(_, dev), i = _
                let px = _A[i, j]
                    _hist[((Int64(getindex(px, 1)) << 16) | (Int64(getindex(px, 2)) << 8) | Int64(getindex(px, 3))) + 1] += 1
                end
            end
        end
    end

    @finch mode = :fast begin
        _hist .= 0
        for j = parallel(_, dev), i = _
            let px = _A[i, j]
                _hist[((Int64(getindex(px, 1)) << 16) | (Int64(getindex(px, 2)) << 8) | Int64(getindex(px, 3))) + 1] += 1
            end
        end
    end

    return (; time=time, hist = _hist)
end

function birdseed_hist_gray(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element(UInt8(0)))), A)
    _hist = Tensor(Coalesce(dev, SparseHash(Element(0), 256)))

    time = @belapsed begin
        (_A, _hist, dev) = $(_A, _hist, dev)
        @finch mode = :fast begin
            _hist .= 0
            for j = parallel(_, dev), i = _
                _hist[_A[i, j]] += 1
            end
        end
    end

    @finch mode = :fast begin
        _hist .= 0
        for j = parallel(_, dev), i = _
            _hist[_A[i, j] + 1] += 1
        end
    end

    return (; time=time, hist = _hist)
end

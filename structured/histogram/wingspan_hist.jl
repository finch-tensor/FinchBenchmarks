using Finch
using BenchmarkTools
using SparseArrays

# In the birdseed Finch, Coalesce can't lower SparseDict (no sample_dims),
# so the kernels are only defined in the wingspan Finch (which has no SparseHash).
if !isdefined(Finch, :SparseHash)
let
    dev = cpu(:t)
    _A = Tensor(Dense(SparseRunList(Element((UInt8(0), UInt8(0), UInt8(0))))))
    # (r, g, b) is packed into a single Int64 key: (r << 16) | (g << 8) | b, 1-based.
    _hist = Tensor(Coalesce(dev, SparseDict(Element(0))))
    eval(@finch_kernel mode = :fast function wingspan_hist_kernel(_hist, _A, dev)
        _hist .= 0
        for j = parallel(_, dev), i = _
            let px = _A[i, j]
                _hist[((Int64(getindex(px, 1)) << 16) | (Int64(getindex(px, 2)) << 8) | Int64(getindex(px, 3))) + 1] += 1
            end
        end
        return _hist
    end)

    _A_gray = Tensor(Dense(SparseRunList(Element(UInt8(0)))))
    _hist_gray = Tensor(Coalesce(dev, SparseDict(Element(0))))
    eval(@finch_kernel mode = :fast function wingspan_hist_gray_kernel(_hist_gray, _A_gray, dev)
        _hist_gray .= 0
        for j = parallel(_, dev), i = _
            _hist_gray[_A_gray[i, j]] += 1
        end
        return _hist_gray
    end)
end
end

function wingspan_hist(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element((UInt8(0), UInt8(0), UInt8(0))))), A)
    _hist = Tensor(Coalesce(dev, SparseDict(Element(0))))

    time = @belapsed wingspan_hist_kernel($_hist, $_A, $dev)
    wingspan_hist_kernel(_hist, _A, dev)

    return (; time=time, hist = _hist)
end

function wingspan_hist_gray(A, num_cpu)
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(SparseRunList(Element(UInt8(0)))), A)
    _hist = Tensor(Coalesce(dev, SparseDict(Element(0))))

    time = @belapsed wingspan_hist_gray_kernel($_hist, $_A, $dev)
    wingspan_hist_gray_kernel(_hist, _A, dev)

    return (; time=time, hist = _hist)
end

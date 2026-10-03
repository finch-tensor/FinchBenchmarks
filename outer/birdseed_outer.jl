using Finch
using BenchmarkTools

# Same as wingspan_outer.jl but the sparse-dict levels are SparseHash levels.
# SparseHash only exists in the birdseed Finch, so the kernel is only defined there.
# C[i,j] = sum_k A[i,k] * B[j,k]  (A, B both M x K, column-major)
if isdefined(Finch, :SparseHash)
let
    z0 = 0.0
    dev = cpu(:t)
    sch = greedy_schedule()
    A = Tensor(SparseHash(SparseHash(Element(z0))))
    BT = Tensor(SparseHash(SparseHash(Element(z0))))
    C = Tensor(Coalesce(dev, SparseHash(SparseHash(Element(z0)))))
    eval(@finch_kernel function birdseed_outer_kernel(C, A, BT, dev, sch)
        C .= 0
        for k=parallel(_, dev, sch), j=_, i=_
            C[i, j] += A[i, k] * BT[j, k]
        end
        return C
    end)
end
end

function birdseed_outer(A, B, nt)
    _A = Tensor(A)
    _B = Tensor(B)
    z = default(_A) * default(_B) + false
    dev = cpu(:t, nt)
    C = Tensor(Coalesce(dev, SparseHash(SparseHash(Element(z)))))
    opt = 999999999
    opt_chk = 1
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for _size in sizes
        sch = greedy_schedule(_size)
        @info "testing with chunk size $_size"
        time = @belapsed birdseed_outer_kernel($C, $_A, $_B, $dev, $sch)
        @info "time: $time"
        if time < opt
            opt = time
            opt_chk = _size
        end
    end
    C_result = Tensor(Coalesce(dev, SparseHash(SparseHash(Element(z)))))
    sch = greedy_schedule(opt_chk)
    C = birdseed_outer_kernel(C_result, _A, _B, dev, sch).C
    return (time = opt, C = C)
end

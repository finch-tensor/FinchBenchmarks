using Finch
using BenchmarkTools

# C[i,j] = sum_k A[i,k] * B[j,k]  (A, B both M x K, column-major)
# In the birdseed Finch, Sparse is SparseDict, which Coalesce can't lower (no sample_dims),
# so the kernel is only defined in the wingspan Finch (which has no SparseHash).
if !isdefined(Finch, :SparseHash)
let
    z0 = 0.0
    dev = cpu(:t)
    sch = greedy_schedule()
    A = Tensor(Dense(SparseList(Element(z0))))
    BT = Tensor(Dense(SparseList(Element(z0))))
    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z0)))))
    eval(@finch_kernel function wingspan_outer_kernel(C, A, BT, dev, sch)
        C .= 0
        for k=parallel(_, dev, sch), j=_, i=_
            C[i, j] += A[i, k] * BT[j, k]
        end
        return C
    end)
end
end

function wingspan_outer(A, B, nt)
    _A = Tensor(A)
    _B = Tensor(B)
    z = default(_A) * default(_B) + false
    dev = cpu(:t, nt)
    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    opt = 999999999
    opt_chk = 1
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for _size in sizes
        sch = greedy_schedule(_size)
        @info "testing with chunk size $_size"
        time = @belapsed wingspan_outer_kernel($C, $_A, $_B, $dev, $sch)
        @info "time: $time"
        if time < opt
            opt = time
            opt_chk = _size
        end
    end
    C_result = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    sch = greedy_schedule(opt_chk)
    C = wingspan_outer_kernel(C_result, _A, _B, dev, sch).C
    return (time = opt, C = C)
end

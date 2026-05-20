using Finch
using BenchmarkTools

for z0 = (0, 0.0, false)
    dev = cpu(:t)
    dev1 = cpu(:t)
    dev2 = cpu(:q)
    
    A = Tensor(Dense(SparseList(Element(z0))))
    B = Tensor(Dense(SparseList(Element(z0))))
    z = default(A) * default(B) + false
    C = Tensor(Dense(Shard(dev, SparseList(Element(z)))))
   

    AT = Tensor(Dense(SparseList(Element(z))))
    BT = Tensor(Dense(SparseList(Element(z))))

    eval(@finch_kernel function spgemm_finch_inner_kernel(C, AT, B, dev)
        C .= 0
        for j=parallel(_, dev), i=_, k=_
            C[i, j] += AT[k, i] * B[k, j]
        end
        return C
    end)

    w = Tensor(SparseByteMap(Element(z)))
    eval(@finch_kernel function spgemm_finch_gustavson_kernel(C, w, A, B, dev)
        C .= 0
        for j=parallel(_, dev)
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
        return C
    end)

    sch = greedy_schedule()
    eval(@finch_kernel function spgemm_finch_dynamic_kernel(C, w, A, B, dev, sch)
        C .= 0
        for j=parallel(_, dev, sch)
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
        return C
    end)

    w = Tensor(Coalesce(dev2, SparseByteMap(Element(z))))
    eval(@finch_kernel function spgemm_finch_nested_kernel(C, w, A, B, dev, dev2, sch)
        C .= 0
        for j=parallel(_, dev, sch)
            w .= 0
            for k=parallel(_, dev2), i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
        return C
    end)

    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    eval(@finch_kernel function spgemm_finch_outer_kernel(C, A, BT, dev)
        C .= 0
        for k=parallel(_, dev), j=_, i=_
            C[i, j] += A[i, k] * BT[j, k]
        end
        return C
    end)

    A = Tensor(Sparse(Sparse(Element(z0))))
    BT = Tensor(Sparse(Sparse(Element(z))))
    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    eval(@finch_kernel function spgemm_finch_outer_kernel(C, A, BT, dev)
        C .= 0
        for k=parallel(_, dev), j=_, i=_
            C[i, j] += A[i, k] * BT[j, k]
        end
        return C
    end)
end

function spgemm_finch_inner_measure(A, B, nt)
    dev = cpu(:t, nt)
    z = default(A) * default(B) + false
    #w2D = Tensor(SparseHash{2}(Element(z)))
    w2D = Tensor(SparseDict(SparseDict(Element(z))))
    AT = Tensor(Dense(SparseList(Element(z))))
    AT = copyto!(AT, swizzle(A, 2, 1))
    C = Tensor(Dense(Shard(dev, SparseList(Element(z)))))
    time = @belapsed spgemm_finch_inner_kernel($C, $AT, $B, $dev)
    C = spgemm_finch_inner_kernel(C, AT, B, dev).C
    return (time = time, C = C)
end

function spgemm_finch_gustavson_measure(A, B, nt)
    z = default(A) * default(B) + false
    dev = cpu(:t, nt)
    C = Tensor(Dense(Shard(dev, SparseList(Element(z)))))
    w = Tensor(SparseByteMap(Element(z)))
    time = @belapsed spgemm_finch_gustavson_kernel($C, $w, $A, $B, $dev)
    C = spgemm_finch_gustavson_kernel(C, w, A, B, dev).C
    return (time = time, C = C)
end

function spgemm_finch_nested_measure(A, B, nt)
    d1_nt = isqrt(nt)
    d2_nt = d1_nt
    z = default(A) * default(B) + false
    dev = cpu(:t, d1_nt)
    dev2 = cpu(:q, d2_nt)
    sch = greedy_schedule()
    C = Tensor(Dense(Shard(dev, SparseList(Element(z)))))
    w = Tensor(Coalesce(dev2, SparseByteMap(Element(z))))
    time = @belapsed spgemm_finch_nested_kernel($C, $w, $A, $B, $dev, $dev2, $sch)
    C = spgemm_finch_nested_kernel(C, w, A, B, dev, dev2, sch).C
    return (time = time, C = C)
end

function spgemm_finch_dynamic_measure(A, B, nt)
    z = default(A) * default(B) + false
    dev = cpu(:t, nt)
    C = Tensor(Dense(Shard(dev, SparseList(Element(z)))))
    w = Tensor(SparseByteMap(Element(z)))

    opt = 999999999
    opt_chk = 1
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for _size in sizes
        sch = greedy_schedule(_size)
        @info "testing with chunk size $_size"
        time = @belapsed spgemm_finch_dynamic_kernel($C, $w, $A, $B, $dev, $sch)
        @info "time: $time"
        if time < opt
            opt = time
            opt_chk = _size
        end
    end

    sch = greedy_schedule(opt_chk)
    C = spgemm_finch_dynamic_kernel(C, w, A, B, dev, sch).C
    return (time = opt, C = C)
end

function spgemm_finch_outer_measure(A, B, nt)
    z = default(A) * default(B) + false
    dev = cpu(:t, nt)
    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    BT = Tensor(Dense(SparseList(Element(z))))
    BT = copyto!(BT, swizzle(B, 2, 1))
    time = @belapsed spgemm_finch_outer_kernel($C, $A, $BT, $dev)
    C_result = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    C = spgemm_finch_outer_kernel(C_result, A, BT, dev).C
    return (time = time, C = C)
end

function spgemm_finch_hypersparse_measure(A, B, nt)
    z = default(A) * default(B) + false
    dev = cpu(:t, nt)
    C = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    A = Tensor(Sparse(Sparse(Element(z))), A)
    BT = Tensor(Sparse(Sparse(Element(z))))
    BT = copyto!(BT, swizzle(B, 2, 1))
    time = @belapsed spgemm_finch_outer_kernel($C, $A, $BT, $dev)
    C_result = Tensor(Coalesce(dev, Sparse(Sparse(Element(z)))))
    C = spgemm_finch_outer_kernel(C_result, A, BT, dev).C
    return (time = time, C = C)
end

function spgemm_finch(f, A, B, nt)
    _A = Tensor(A)
    _B = Tensor(B)
    C = Ref{Any}()
    (time, C[]) = f(_A, _B, nt)
    return (;time = time, C = C[])
end

spgemm_finch_inner(A, B, nt) = spgemm_finch(spgemm_finch_inner_measure, A, B, nt)
spgemm_finch_gustavson(A, B, nt) = spgemm_finch(spgemm_finch_gustavson_measure, A, B, nt)
spgemm_finch_outer(A, B, nt) = spgemm_finch(spgemm_finch_outer_measure, A, B, nt)
spgemm_finch_hypersparse(A, B, nt) = spgemm_finch(spgemm_finch_hypersparse_measure, A, B, nt)
spgemm_finch_dynamic(A, B, nt) = spgemm_finch(spgemm_finch_dynamic_measure, A, B, nt)
spgemm_finch_nested(A, B, nt) = spgemm_finch(spgemm_finch_nested_measure, A, B, nt)
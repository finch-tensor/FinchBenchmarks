using Finch
using BenchmarkTools
using SuiteSparseGraphBLAS

function coalesce_spmspv(A, x, nt)
        dev = cpu(:t, nt)
        _y = Tensor(Coalesce(dev, SparseByteMap(Element(0.0))))
        _x = Tensor(SparseList(Element(0.0)), x)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        time = @belapsed begin
                (_A, _x, _y, dev) = $(_A, _x, _y, dev)
                @finch mode = :fast begin
                        _y .= 0
                        for j = parallel(_, dev), i = _
                                _y[i] += _A[i, j] * _x[j]
                        end
                end
        end

        @finch mode = :fast begin
                        _y .= 0
                        for j = parallel(_, dev), i = _
                                _y[i] += _A[i, j] * _x[j]
                        end
                end
        return (; time=time, y=_y)
end


function coalesce_spmspv_dynamic(A, x, nt)
        dev = cpu(:t, nt)
        _y = Tensor(Coalesce(dev, SparseDict(Element(0.0))))
        _x = Tensor(SparseList(Element(0.0)), x)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)

        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        opt = 99999999
        opt_chk = 0
        for size in sizes
                @info "testing with chunk $size"
                sch = greedy_schedule(size)
                time = @belapsed begin
                        (_A, _x, _y, dev, sch) = $(_A, _x, _y, dev, sch)
                        @finch mode = :fast begin
                                _y .= 0
                                for j = parallel(_, dev, sch), i = _
                                        _y[i] += _A[i, j] * _x[j]
                                end
                        end
                end
                @info "time: $time"
                if time < opt
                        opt = time
                        opt_chk = size
                end
        end

        sch = greedy_schedule(opt_chk)
        @finch mode = :fast begin
                _y .= 0
                for j = parallel(_, dev, sch), i = _
                        _y[i] += _A[i, j] * _x[j]
                end
        end
        return (; time=opt, y=_y)
end

function blas_spmspv(A, x, nt)
        _A = GBMatrix(A)
        idx, vals = findnz(x)
        _x = GBVector(idx, vals, length(x))

        time = @belapsed $_A * $_x
        y = _A * _x

        return (; time=time, y=y)
end
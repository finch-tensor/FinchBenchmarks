using Finch
using BenchmarkTools

y = Tensor(Dense(Element(0.0)))
x = Tensor(Dense(Element(0.0)))
y_j = Scalar(0.0)

for (A, diag) in [
    (Tensor(Dense(SparseList(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparseList(Pattern()))), Tensor(Dense(Element(false)))),
    (Tensor(Dense(SparseVBLLevel(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparseBand(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparsePoint(Pattern()))), Tensor(Dense(Element(false))))
]
    eval(@finch_kernel mode=:fast function spmv_finch_sym_kernel(y, A, x, diag, y_j)
        y .= 0
        for j = _
            let x_j = x[j]
                y_j .= 0
                for i = _
                    let A_ij = A[i, j]
                        y[i] += x_j * A_ij
                        y_j[] += A_ij * x[i]
                    end
                end
                y[j] += y_j[] + diag[j] * x_j
            end
        end
        return y
    end)

    eval(@finch_kernel mode=:fast function spmv_finch_col_maj_kernel(y, A, x)
        y .= 0
        for j = _, i = _
            y[i] += A[i, j] * x[j]
        end
        return y
    end)

    eval(@finch_kernel mode=:fast function spmv_finch_row_maj_kernel(y, A, x)
        y .= 0
        for j = _, i = _
            y[j] += A[i, j] * x[i]
        end
        return y
    end)
end


function spmv_finch_sym_sparselist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    _y_j = Scalar(0.0)
    _x = Tensor(Dense(Element(0.0)), x)

    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_sym_kernel($_y, $_A, $_x, $_d, $y_j).y
    return (;time = time, y = y[])
end

function spmv_finch_col_maj_sparselist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_col_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_row_maj_sparselist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_row_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_sym_sparseband(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    _y_j = Scalar(0.0)
    _x = Tensor(Dense(Element(0.0)), x)

    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_sym_kernel($_y, $_A, $_x, $_d, $y_j).y
    return (;time = time, y = y[])
end

function spmv_finch_col_maj_sparseband(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_col_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_row_maj_sparseband(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_row_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_sym_sparselist_pattern(y, A, x) 
    A = pattern!(Tensor(CSCFormat(), A))
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Pattern())), A)
    _d = Tensor(Dense(Element(false)))
    _y_j = Scalar(0.0)
    _x = Tensor(Dense(Element(0.0)), x)

    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_sym_kernel($_y, $_A, $_x, $_d, $y_j).y
    return (;time = time, y = y[])
end

function spmv_finch_col_maj_sparselist_pattern(y, A, x) 
    A = pattern!(Tensor(CSCFormat(), A))
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Pattern())), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_col_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_row_maj_sparselist_pattern(y, A, x) 
    A = pattern!(Tensor(CSCFormat(), A))
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Pattern())), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_row_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_col_maj_sparsepoint_pattern(y, A, x) 
    A = pattern!(Tensor(CSCFormat(), A))
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Pattern())), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_col_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_row_maj_sparsepoint_pattern(y, A, x) 
    A = pattern!(Tensor(CSCFormat(), A))
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Pattern())), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_row_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_sym_sparseblocklist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBlockList(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    _y_j = Scalar(0.0)
    _x = Tensor(Dense(Element(0.0)), x)

    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_sym_kernel($_y, $_A, $_x, $_d, $y_j).y
    return (;time = time, y = y[])
end

function spmv_finch_col_maj_sparseblocklist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBlockList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_col_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end

function spmv_finch_row_maj_sparseblocklist(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBlockList(Element(0.0))), permutedims(A))
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_row_maj_kernel($_y, $_A, $_x).y
    return (;time = time, y = y[])
end
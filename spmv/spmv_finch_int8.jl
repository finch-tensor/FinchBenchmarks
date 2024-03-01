using Finch
using BenchmarkTools

function ssymv_finch_int8_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int8, Int64, Vector{Int8}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0, Int8, Int64, Vector{Int8}}}}, y_j::Scalar{0.0, Float64})
    @inbounds begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            y_lvl_q = (1 - 1) * A_lvl.shape + A_lvl_2_i
                            x_lvl_q_2 = (1 - 1) * x_lvl.shape + A_lvl_2_i
                            x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                            y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                            y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                y_lvl_q = (1 - 1) * A_lvl.shape + phase_stop_3
                                x_lvl_q_2 = (1 - 1) * x_lvl.shape + phase_stop_3
                                x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                y_j_val += A_lvl_3_val * x_lvl_2_val_3
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_int8_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_int8_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_int8(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(Int8(0)))), A)
    _d = Tensor(Dense(Element(Int8(0))))
    @finch begin
        _A .= Int8(0)
        _d .= Int8(0)
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)

    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_int8_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
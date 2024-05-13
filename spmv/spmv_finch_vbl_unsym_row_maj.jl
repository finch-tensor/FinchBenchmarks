using Finch
using BenchmarkTools

function spmv_finch_vbl_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_vbl::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_vbl_lvl = A_vbl.lvl
            A_vbl_lvl_2 = A_vbl_lvl.lvl
            A_vbl_lvl_ptr = A_vbl_lvl_2.ptr
            A_vbl_lvl_idx = A_vbl_lvl_2.idx
            A_vbl_lvl_ofs = A_vbl_lvl_2.ofs
            A_vbl_lvl_2_val = A_vbl_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_vbl_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_vbl_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_vbl_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_vbl_lvl.shape)
            for j_4 = 1:A_vbl_lvl.shape
                y_lvl_q = (1 - 1) * A_vbl_lvl.shape + j_4
                A_vbl_lvl_q = (1 - 1) * A_vbl_lvl.shape + j_4
                A_vbl_lvl_2_r = A_vbl_lvl_ptr[A_vbl_lvl_q]
                A_vbl_lvl_2_r_stop = A_vbl_lvl_ptr[A_vbl_lvl_q + 1]
                if A_vbl_lvl_2_r < A_vbl_lvl_2_r_stop
                    A_vbl_lvl_2_i1 = A_vbl_lvl_idx[A_vbl_lvl_2_r_stop - 1]
                else
                    A_vbl_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_vbl_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_vbl_lvl_idx[A_vbl_lvl_2_r] < 1
                        A_vbl_lvl_2_r = Finch.scansearch(A_vbl_lvl_idx, 1, A_vbl_lvl_2_r, A_vbl_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_vbl_lvl_2_i = A_vbl_lvl_idx[A_vbl_lvl_2_r]
                        A_vbl_lvl_2_q_stop = A_vbl_lvl_ofs[A_vbl_lvl_2_r + 1]
                        A_vbl_lvl_2_i_2 = A_vbl_lvl_2_i - (A_vbl_lvl_2_q_stop - A_vbl_lvl_ofs[A_vbl_lvl_2_r])
                        A_vbl_lvl_2_q_ofs = (A_vbl_lvl_2_q_stop - A_vbl_lvl_2_i) - 1
                        if A_vbl_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_vbl_lvl_2_i_2)
                            if A_vbl_lvl_2_i >= phase_start_3
                                for i_8 = phase_start_3:A_vbl_lvl_2_i
                                    x_lvl_q = (1 - 1) * x_lvl.shape + i_8
                                    A_vbl_lvl_2_q = A_vbl_lvl_2_q_ofs + i_8
                                    x_lvl_2_val = x_lvl_val[x_lvl_q]
                                    A_vbl_lvl_3_val = A_vbl_lvl_2_val[A_vbl_lvl_2_q]
                                    y_lvl_val[y_lvl_q] += A_vbl_lvl_3_val * x_lvl_2_val
                                end
                            end
                            A_vbl_lvl_2_r += A_vbl_lvl_2_i == A_vbl_lvl_2_i
                            i = A_vbl_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_vbl_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_vbl_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_11 = phase_start_6:phase_stop_5
                                    x_lvl_q = (1 - 1) * x_lvl.shape + i_11
                                    A_vbl_lvl_2_q = A_vbl_lvl_2_q_ofs + i_11
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q]
                                    A_vbl_lvl_3_val_2 = A_vbl_lvl_2_val[A_vbl_lvl_2_q]
                                    y_lvl_val[y_lvl_q] += A_vbl_lvl_3_val_2 * x_lvl_2_val_2
                                end
                            end
                            A_vbl_lvl_2_r += phase_stop_5 == A_vbl_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_vbl_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_vbl_lvl.shape)),)
        end
end

function spmv_finch_vbl_kernel_row_maj(y, A, x)
    spmv_finch_vbl_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_vbl_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_vbl_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end

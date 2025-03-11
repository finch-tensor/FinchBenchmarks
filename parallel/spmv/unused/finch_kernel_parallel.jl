using Finch
using BenchmarkTools

function finch_kernel_parallel_helper(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
    @inbounds @fastmath(begin
        y_lvl = y.lvl
        y_lvl_val = y_lvl.lvl.val
        tns_lvl = A.body.lvl
        tns_lvl_2 = tns_lvl.lvl
        tns_lvl_ptr = tns_lvl_2.ptr
        tns_lvl_idx = tns_lvl_2.idx
        tns_lvl_2_val = tns_lvl_2.lvl.val
        x_lvl = x.lvl
        x_lvl_val = x_lvl.lvl.val
        x_lvl.shape == tns_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(tns_lvl_2.shape))"))
        Finch.resize_if_smaller!(y_lvl_val, tns_lvl.shape)
        Finch.fill_range!(y_lvl_val, 0.0, 1, tns_lvl.shape)
        val = y_lvl_val
        y_lvl_val = (Finch).moveto(y_lvl_val, CPU(Threads.nthreads()))
        x_lvl_val = (Finch).moveto(x_lvl_val, CPU(Threads.nthreads()))
        tns_lvl_ptr = (Finch).moveto(tns_lvl_ptr, CPU(Threads.nthreads()))
        tns_lvl_idx = (Finch).moveto(tns_lvl_idx, CPU(Threads.nthreads()))
        tns_lvl_2_val = (Finch).moveto(tns_lvl_2_val, CPU(Threads.nthreads()))
        Threads.@threads for i_4 = 1:Threads.nthreads()
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    phase_start_2 = max(1, 1 + fld(tns_lvl.shape * (i_4 + -1), Threads.nthreads()))
                    phase_stop_2 = min(tns_lvl.shape, fld(tns_lvl.shape * i_4, Threads.nthreads()))
                    if phase_stop_2 >= phase_start_2
                        for i_7 = phase_start_2:phase_stop_2
                            y_lvl_q = (1 - 1) * tns_lvl.shape + i_7
                            tns_lvl_q = (1 - 1) * tns_lvl.shape + i_7
                            tns_lvl_2_q = tns_lvl_ptr[tns_lvl_q]
                            tns_lvl_2_q_stop = tns_lvl_ptr[tns_lvl_q+1]
                            if tns_lvl_2_q < tns_lvl_2_q_stop
                                tns_lvl_2_i1 = tns_lvl_idx[tns_lvl_2_q_stop-1]
                            else
                                tns_lvl_2_i1 = 0
                            end
                            phase_stop_3 = min(x_lvl.shape, tns_lvl_2_i1)
                            if phase_stop_3 >= 1
                                if tns_lvl_idx[tns_lvl_2_q] < 1
                                    tns_lvl_2_q = Finch.scansearch(tns_lvl_idx, 1, tns_lvl_2_q, tns_lvl_2_q_stop - 1)
                                end
                                while true
                                    tns_lvl_2_i = tns_lvl_idx[tns_lvl_2_q]
                                    if tns_lvl_2_i < phase_stop_3
                                        tns_lvl_3_val = tns_lvl_2_val[tns_lvl_2_q]
                                        x_lvl_q = (1 - 1) * x_lvl.shape + tns_lvl_2_i
                                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                                        y_lvl_val[y_lvl_q] = tns_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                        tns_lvl_2_q += 1
                                    else
                                        phase_stop_5 = min(phase_stop_3, tns_lvl_2_i)
                                        if tns_lvl_2_i == phase_stop_5
                                            tns_lvl_3_val = tns_lvl_2_val[tns_lvl_2_q]
                                            x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop_5
                                            x_lvl_2_val_2 = x_lvl_val[x_lvl_q]
                                            y_lvl_val[y_lvl_q] += tns_lvl_3_val * x_lvl_2_val_2
                                            tns_lvl_2_q += 1
                                        end
                                        break
                                    end
                                end
                            end
                        end
                    end
                    phase_start_6 = max(1, 1 + fld(tns_lvl.shape * i_4, Threads.nthreads()))
                    phase_stop_7 = tns_lvl.shape
                    if phase_stop_7 >= phase_start_6
                        phase_stop_7 + 1
                    end
                end)
                nothing
            end
        end
        resize!(val, tns_lvl.shape)
    end)
end

function finch_kernel_parallel(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)

    time = @belapsed finch_kernel_parallel_helper($_y, $_A, $_x) 
    return (; time=time, y=_y)
end

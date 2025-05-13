using Finch
using BenchmarkTools

function spgemm_finch_kernel_static_gustavson_dense_helper(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, C::Tensor{DenseLevel{Int64,DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, w::Tensor{SparseByteMapLevel{Int64,Vector{Int64},Vector{Bool},Vector{Tuple{Int64,Int64}},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
    @inbounds @fastmath(begin
        A_lvl = A.lvl
        A_lvl_stop = A_lvl.shape
        A_lvl_2 = A_lvl.lvl
        A_lvl_2_ptr = A_lvl_2.ptr
        A_lvl_2_idx = A_lvl_2.idx
        A_lvl_2_stop = A_lvl_2.shape
        A_lvl_3 = A_lvl_2.lvl
        A_lvl_3_val = A_lvl_3.val
        B_lvl = B.lvl
        B_lvl_stop = B_lvl.shape
        B_lvl_2 = B_lvl.lvl
        B_lvl_2_ptr = B_lvl_2.ptr
        B_lvl_2_idx = B_lvl_2.idx
        B_lvl_2_stop = B_lvl_2.shape
        B_lvl_3 = B_lvl_2.lvl
        B_lvl_3_val = B_lvl_3.val
        C_lvl = C.lvl
        C_lvl_2 = C_lvl.lvl
        C_lvl_3 = C_lvl_2.lvl
        C_lvl_3_val = C_lvl_3.val
        w_lvl = w.lvl
        w_lvl_ptr = w_lvl.ptr
        w_lvl_tbl = w_lvl.tbl
        w_lvl_srt = w_lvl.srt
        w_lvl_qos_stop = (w_lvl_qos_fill = length(w_lvl.srt))
        w_lvl_2 = w_lvl.lvl
        w_lvl_2_val = w_lvl_2.val
        B_lvl_2_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(B_lvl_2_stop) != $(A_lvl_stop))"))
        pos_stop = A_lvl_2_stop * B_lvl_stop
        Finch.resize_if_smaller!(C_lvl_3_val, pos_stop)
        Finch.fill_range!(C_lvl_3_val, 0.0, 1, pos_stop)
        w_lvl_2_val_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_2_val)
        w_lvl_ptr_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_ptr)
        w_lvl_tbl_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_tbl)
        w_lvl_srt_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_srt)
        B_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_3_val)
        B_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_2_ptr)
        B_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_2_idx)
        A_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_3_val)
        A_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_2_ptr)
        A_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_2_idx)
        C_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), C_lvl_3_val)
        Threads.@threads :dynamic for tid = 1:1
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    w_lvl_2_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_2_val_2)
                    w_lvl_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_ptr_2)
                    w_lvl_tbl_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_tbl_2)
                    w_lvl_srt_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_srt_2)
                    B_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_3_val_2)
                    B_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_2_ptr_2)
                    B_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_2_idx_2)
                    A_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_3_val_2)
                    A_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_2_ptr_2)
                    A_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_2_idx_2)
                    C_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), C_lvl_3_val_2)
                    phase_start_2 = max(1, 1 + fld(B_lvl_stop * (tid + -1), 1))
                    phase_stop_2 = min(B_lvl_stop, fld(B_lvl_stop * tid, 1))
                    if phase_stop_2 >= phase_start_2
                        for j_6 = phase_start_2:phase_stop_2
                            B_lvl_q = (1 - 1) * B_lvl_stop + j_6
                            C_lvl_q = (1 - 1) * B_lvl_stop + j_6
                            for w_lvl_r = 1:w_lvl_qos_fill
                                w_lvl_p = first(w_lvl_srt_3[w_lvl_r])
                                w_lvl_ptr_3[w_lvl_p] = 0
                                w_lvl_ptr_3[w_lvl_p+1] = 0
                                w_lvl_i = last(w_lvl_srt_3[w_lvl_r])
                                w_lvl_q = (w_lvl_p - 1) * A_lvl_2_stop + w_lvl_i
                                w_lvl_tbl_3[w_lvl_q] = false
                                Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvl_q)
                                Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvl_q, w_lvl_q)
                            end
                            w_lvl_qos_fill = 0
                            w_lvl_ptr_3[1] = 1
                            w_lvlq_stop = 1A_lvl_2_stop
                            Finch.resize_if_smaller!(w_lvl_ptr_3, 1 + 1)
                            Finch.fill_range!(w_lvl_ptr_3, 0, 1 + 1, 1 + 1)
                            w_lvlold = length(w_lvl_tbl_3) + 1
                            Finch.resize_if_smaller!(w_lvl_tbl_3, w_lvlq_stop)
                            Finch.fill_range!(w_lvl_tbl_3, false, w_lvlold, w_lvlq_stop)
                            Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvlq_stop)
                            Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvlold, w_lvlq_stop)
                            B_lvl_2_q = B_lvl_2_ptr_3[B_lvl_q]
                            B_lvl_2_q_stop = B_lvl_2_ptr_3[B_lvl_q+1]
                            if B_lvl_2_q < B_lvl_2_q_stop
                                B_lvl_2_i1 = B_lvl_2_idx_3[B_lvl_2_q_stop-1]
                            else
                                B_lvl_2_i1 = 0
                            end
                            phase_stop_3 = min(B_lvl_2_stop, B_lvl_2_i1)
                            if phase_stop_3 >= 1
                                if B_lvl_2_idx_3[B_lvl_2_q] < 1
                                    B_lvl_2_q = Finch.scansearch(B_lvl_2_idx_3, 1, B_lvl_2_q, B_lvl_2_q_stop - 1)
                                end
                                while true
                                    B_lvl_2_i = B_lvl_2_idx_3[B_lvl_2_q]
                                    if B_lvl_2_i < phase_stop_3
                                        B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                        A_lvl_q = (1 - 1) * A_lvl_stop + B_lvl_2_i
                                        A_lvl_2_q = A_lvl_2_ptr_3[A_lvl_q]
                                        A_lvl_2_q_stop = A_lvl_2_ptr_3[A_lvl_q+1]
                                        if A_lvl_2_q < A_lvl_2_q_stop
                                            A_lvl_2_i1 = A_lvl_2_idx_3[A_lvl_2_q_stop-1]
                                        else
                                            A_lvl_2_i1 = 0
                                        end
                                        phase_stop_5 = min(A_lvl_2_stop, A_lvl_2_i1)
                                        if phase_stop_5 >= 1
                                            if A_lvl_2_idx_3[A_lvl_2_q] < 1
                                                A_lvl_2_q = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                                            end
                                            while true
                                                A_lvl_2_i = A_lvl_2_idx_3[A_lvl_2_q]
                                                if A_lvl_2_i < phase_stop_5
                                                    A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                    w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i
                                                    w_lvl_2_val_3[w_lvl_q_2] = B_lvl_3_val_4 * A_lvl_3_val_4 + w_lvl_2_val_3[w_lvl_q_2]
                                                    if !(w_lvl_tbl_3[w_lvl_q_2])
                                                        w_lvl_tbl_3[w_lvl_q_2] = true
                                                        w_lvl_qos_fill += 1
                                                        if w_lvl_qos_fill > w_lvl_qos_stop
                                                            w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                            Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                        end
                                                        w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i)
                                                    end
                                                    A_lvl_2_q += 1
                                                else
                                                    phase_stop_7 = min(phase_stop_5, A_lvl_2_i)
                                                    if A_lvl_2_i == phase_stop_7
                                                        A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                        w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + phase_stop_7
                                                        w_lvl_2_val_3[w_lvl_q_2] += B_lvl_3_val_4 * A_lvl_3_val_4
                                                        if !(w_lvl_tbl_3[w_lvl_q_2])
                                                            w_lvl_tbl_3[w_lvl_q_2] = true
                                                            w_lvl_qos_fill += 1
                                                            if w_lvl_qos_fill > w_lvl_qos_stop
                                                                w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                            end
                                                            w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_7)
                                                        end
                                                        A_lvl_2_q += 1
                                                    end
                                                    break
                                                end
                                            end
                                        end
                                        B_lvl_2_q += 1
                                    else
                                        phase_stop_9 = min(phase_stop_3, B_lvl_2_i)
                                        if B_lvl_2_i == phase_stop_9
                                            B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                            A_lvl_q = (1 - 1) * A_lvl_stop + phase_stop_9
                                            A_lvl_2_q_2 = A_lvl_2_ptr_3[A_lvl_q]
                                            A_lvl_2_q_stop_2 = A_lvl_2_ptr_3[A_lvl_q+1]
                                            if A_lvl_2_q_2 < A_lvl_2_q_stop_2
                                                A_lvl_2_i1_2 = A_lvl_2_idx_3[A_lvl_2_q_stop_2-1]
                                            else
                                                A_lvl_2_i1_2 = 0
                                            end
                                            phase_stop_10 = min(A_lvl_2_stop, A_lvl_2_i1_2)
                                            if phase_stop_10 >= 1
                                                if A_lvl_2_idx_3[A_lvl_2_q_2] < 1
                                                    A_lvl_2_q_2 = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q_2, A_lvl_2_q_stop_2 - 1)
                                                end
                                                while true
                                                    A_lvl_2_i_2 = A_lvl_2_idx_3[A_lvl_2_q_2]
                                                    if A_lvl_2_i_2 < phase_stop_10
                                                        A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                        w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i_2
                                                        w_lvl_2_val_3[w_lvl_q_3] = B_lvl_3_val_4 * A_lvl_3_val_5 + w_lvl_2_val_3[w_lvl_q_3]
                                                        if !(w_lvl_tbl_3[w_lvl_q_3])
                                                            w_lvl_tbl_3[w_lvl_q_3] = true
                                                            w_lvl_qos_fill += 1
                                                            if w_lvl_qos_fill > w_lvl_qos_stop
                                                                w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                            end
                                                            w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i_2)
                                                        end
                                                        A_lvl_2_q_2 += 1
                                                    else
                                                        phase_stop_12 = min(phase_stop_10, A_lvl_2_i_2)
                                                        if A_lvl_2_i_2 == phase_stop_12
                                                            A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                            w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + phase_stop_12
                                                            w_lvl_2_val_3[w_lvl_q_3] += B_lvl_3_val_4 * A_lvl_3_val_5
                                                            if !(w_lvl_tbl_3[w_lvl_q_3])
                                                                w_lvl_tbl_3[w_lvl_q_3] = true
                                                                w_lvl_qos_fill += 1
                                                                if w_lvl_qos_fill > w_lvl_qos_stop
                                                                    w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                    Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                end
                                                                w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_12)
                                                            end
                                                            A_lvl_2_q_2 += 1
                                                        end
                                                        break
                                                    end
                                                end
                                            end
                                            B_lvl_2_q += 1
                                        end
                                        break
                                    end
                                end
                            end
                            resize!(w_lvl_ptr_3, 1 + 1)
                            resize!(w_lvl_tbl_3, 1A_lvl_2_stop)
                            resize!(w_lvl_srt_3, w_lvl_qos_fill)
                            sort!(w_lvl_srt_3)
                            w_lvl_p_prev = 0
                            for w_lvl_r_2 = 1:w_lvl_qos_fill
                                w_lvl_p_2 = first(w_lvl_srt_3[w_lvl_r_2])
                                if w_lvl_p_2 != w_lvl_p_prev
                                    w_lvl_ptr_3[w_lvl_p_prev+1] = w_lvl_r_2
                                    w_lvl_ptr_3[w_lvl_p_2] = w_lvl_r_2
                                end
                                w_lvl_p_prev = w_lvl_p_2
                            end
                            w_lvl_ptr_3[w_lvl_p_prev+1] = w_lvl_qos_fill + 1
                            w_lvl_qos_stop = w_lvl_qos_fill
                            resize!(w_lvl_2_val_3, A_lvl_2_stop)
                            w_lvl_r_3 = w_lvl_ptr_3[1]
                            w_lvl_r_stop = w_lvl_ptr_3[1+1]
                            if w_lvl_r_3 != 0 && w_lvl_r_3 < w_lvl_r_stop
                                w_lvl_i_stop = last(w_lvl_srt_3[w_lvl_r_stop-1])
                            else
                                w_lvl_i_stop = 0
                            end
                            phase_stop_15 = min(A_lvl_2_stop, w_lvl_i_stop)
                            if phase_stop_15 >= 1
                                while w_lvl_r_3 + 1 < w_lvl_r_stop && last(w_lvl_srt_3[w_lvl_r_3]) < 1
                                    w_lvl_r_3 += 1
                                end
                                while true
                                    w_lvl_i_2 = last(w_lvl_srt_3[w_lvl_r_3])
                                    if w_lvl_i_2 < phase_stop_15
                                        w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                        w_lvl_2_val_4 = w_lvl_2_val_3[w_lvl_q_4]
                                        C_lvl_2_q = (C_lvl_q - 1) * A_lvl_2_stop + w_lvl_i_2
                                        C_lvl_3_val_3[C_lvl_2_q] = w_lvl_2_val_4
                                        w_lvl_r_3 += 1
                                    else
                                        phase_stop_17 = min(phase_stop_15, w_lvl_i_2)
                                        if w_lvl_i_2 == phase_stop_17
                                            w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                            w_lvl_2_val_5 = w_lvl_2_val_3[w_lvl_q_4]
                                            C_lvl_2_q = (C_lvl_q - 1) * A_lvl_2_stop + phase_stop_17
                                            C_lvl_3_val_3[C_lvl_2_q] = w_lvl_2_val_5
                                            w_lvl_r_3 += 1
                                        end
                                        break
                                    end
                                end
                            end
                        end
                    end
                    phase_start_15 = max(1, 1 + fld(B_lvl_stop * tid, 1))
                    if B_lvl_stop >= phase_start_15
                        B_lvl_stop + 1
                    end
                end)
                nothing
            end
        end
        resize!(C_lvl_3_val_2, A_lvl_2_stop * B_lvl_stop)
        Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0.0,Float64,Int64}(C_lvl_3_val_2), A_lvl_2_stop), B_lvl_stop))
    end)
end

function spgemm_finch_kernel_greedy_gustavson_dense_helper(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, C::Tensor{DenseLevel{Int64,DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, w::Tensor{SparseByteMapLevel{Int64,Vector{Int64},Vector{Bool},Vector{Tuple{Int64,Int64}},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
    @inbounds @fastmath(begin
        A_lvl = A.lvl
        A_lvl_stop = A_lvl.shape
        A_lvl_2 = A_lvl.lvl
        A_lvl_2_ptr = A_lvl_2.ptr
        A_lvl_2_idx = A_lvl_2.idx
        A_lvl_2_stop = A_lvl_2.shape
        A_lvl_3 = A_lvl_2.lvl
        A_lvl_3_val = A_lvl_3.val
        B_lvl = B.lvl
        B_lvl_stop = B_lvl.shape
        B_lvl_2 = B_lvl.lvl
        B_lvl_2_ptr = B_lvl_2.ptr
        B_lvl_2_idx = B_lvl_2.idx
        B_lvl_2_stop = B_lvl_2.shape
        B_lvl_3 = B_lvl_2.lvl
        B_lvl_3_val = B_lvl_3.val
        C_lvl = C.lvl
        C_lvl_2 = C_lvl.lvl
        C_lvl_3 = C_lvl_2.lvl
        C_lvl_3_val = C_lvl_3.val
        w_lvl = w.lvl
        w_lvl_ptr = w_lvl.ptr
        w_lvl_tbl = w_lvl.tbl
        w_lvl_srt = w_lvl.srt
        w_lvl_qos_stop = (w_lvl_qos_fill = length(w_lvl.srt))
        w_lvl_2 = w_lvl.lvl
        w_lvl_2_val = w_lvl_2.val
        B_lvl_2_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(B_lvl_2_stop) != $(A_lvl_stop))"))
        pos_stop = A_lvl_2_stop * B_lvl_stop
        Finch.resize_if_smaller!(C_lvl_3_val, pos_stop)
        Finch.fill_range!(C_lvl_3_val, 0.0, 1, pos_stop)
        w_lvl_2_val_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_2_val)
        w_lvl_ptr_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_ptr)
        w_lvl_tbl_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_tbl)
        w_lvl_srt_2 = (Finch).transfer(Finch.CPULocalMemory(Finch.CPU(1)), w_lvl_srt)
        B_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_3_val)
        B_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_2_ptr)
        B_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), B_lvl_2_idx)
        A_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_3_val)
        A_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_2_ptr)
        A_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), A_lvl_2_idx)
        C_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(1)), C_lvl_3_val)
        chk_ctr = Threads.Atomic{Int}(0)
        num_chks = cld(B_lvl_stop, 100)
        Threads.@threads :static for tid = 1:1
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    w_lvl_2_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_2_val_2)
                    w_lvl_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_ptr_2)
                    w_lvl_tbl_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_tbl_2)
                    w_lvl_srt_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), w_lvl_srt_2)
                    B_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_3_val_2)
                    B_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_2_ptr_2)
                    B_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), B_lvl_2_idx_2)
                    A_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_3_val_2)
                    A_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_2_ptr_2)
                    A_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), A_lvl_2_idx_2)
                    C_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(1), Finch.Serial()), C_lvl_3_val_2)
                    while true
                        chk_id = Threads.atomic_add!(chk_ctr, 1)
                        if chk_id > num_chks
                            break
                        end
                        phase_start_2 = max(1, 1 + 100 * (-1 + chk_id))
                        phase_stop_2 = min(B_lvl_stop, 100chk_id)
                        if phase_stop_2 >= phase_start_2
                            for j_6 = phase_start_2:phase_stop_2
                                B_lvl_q = (1 - 1) * B_lvl_stop + j_6
                                C_lvl_q = (1 - 1) * B_lvl_stop + j_6
                                for w_lvl_r = 1:w_lvl_qos_fill
                                    w_lvl_p = first(w_lvl_srt_3[w_lvl_r])
                                    w_lvl_ptr_3[w_lvl_p] = 0
                                    w_lvl_ptr_3[w_lvl_p+1] = 0
                                    w_lvl_i = last(w_lvl_srt_3[w_lvl_r])
                                    w_lvl_q = (w_lvl_p - 1) * A_lvl_2_stop + w_lvl_i
                                    w_lvl_tbl_3[w_lvl_q] = false
                                    Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvl_q)
                                    Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvl_q, w_lvl_q)
                                end
                                w_lvl_qos_fill = 0
                                w_lvl_ptr_3[1] = 1
                                w_lvlq_stop = 1A_lvl_2_stop
                                Finch.resize_if_smaller!(w_lvl_ptr_3, 1 + 1)
                                Finch.fill_range!(w_lvl_ptr_3, 0, 1 + 1, 1 + 1)
                                w_lvlold = length(w_lvl_tbl_3) + 1
                                Finch.resize_if_smaller!(w_lvl_tbl_3, w_lvlq_stop)
                                Finch.fill_range!(w_lvl_tbl_3, false, w_lvlold, w_lvlq_stop)
                                Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvlq_stop)
                                Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvlold, w_lvlq_stop)
                                B_lvl_2_q = B_lvl_2_ptr_3[B_lvl_q]
                                B_lvl_2_q_stop = B_lvl_2_ptr_3[B_lvl_q+1]
                                if B_lvl_2_q < B_lvl_2_q_stop
                                    B_lvl_2_i1 = B_lvl_2_idx_3[B_lvl_2_q_stop-1]
                                else
                                    B_lvl_2_i1 = 0
                                end
                                phase_stop_3 = min(B_lvl_2_stop, B_lvl_2_i1)
                                if phase_stop_3 >= 1
                                    if B_lvl_2_idx_3[B_lvl_2_q] < 1
                                        B_lvl_2_q = Finch.scansearch(B_lvl_2_idx_3, 1, B_lvl_2_q, B_lvl_2_q_stop - 1)
                                    end
                                    while true
                                        B_lvl_2_i = B_lvl_2_idx_3[B_lvl_2_q]
                                        if B_lvl_2_i < phase_stop_3
                                            B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                            A_lvl_q = (1 - 1) * A_lvl_stop + B_lvl_2_i
                                            A_lvl_2_q = A_lvl_2_ptr_3[A_lvl_q]
                                            A_lvl_2_q_stop = A_lvl_2_ptr_3[A_lvl_q+1]
                                            if A_lvl_2_q < A_lvl_2_q_stop
                                                A_lvl_2_i1 = A_lvl_2_idx_3[A_lvl_2_q_stop-1]
                                            else
                                                A_lvl_2_i1 = 0
                                            end
                                            phase_stop_5 = min(A_lvl_2_stop, A_lvl_2_i1)
                                            if phase_stop_5 >= 1
                                                if A_lvl_2_idx_3[A_lvl_2_q] < 1
                                                    A_lvl_2_q = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                                                end
                                                while true
                                                    A_lvl_2_i = A_lvl_2_idx_3[A_lvl_2_q]
                                                    if A_lvl_2_i < phase_stop_5
                                                        A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                        w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i
                                                        w_lvl_2_val_3[w_lvl_q_2] = B_lvl_3_val_4 * A_lvl_3_val_4 + w_lvl_2_val_3[w_lvl_q_2]
                                                        if !(w_lvl_tbl_3[w_lvl_q_2])
                                                            w_lvl_tbl_3[w_lvl_q_2] = true
                                                            w_lvl_qos_fill += 1
                                                            if w_lvl_qos_fill > w_lvl_qos_stop
                                                                w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                            end
                                                            w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i)
                                                        end
                                                        A_lvl_2_q += 1
                                                    else
                                                        phase_stop_7 = min(phase_stop_5, A_lvl_2_i)
                                                        if A_lvl_2_i == phase_stop_7
                                                            A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                            w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + phase_stop_7
                                                            w_lvl_2_val_3[w_lvl_q_2] += B_lvl_3_val_4 * A_lvl_3_val_4
                                                            if !(w_lvl_tbl_3[w_lvl_q_2])
                                                                w_lvl_tbl_3[w_lvl_q_2] = true
                                                                w_lvl_qos_fill += 1
                                                                if w_lvl_qos_fill > w_lvl_qos_stop
                                                                    w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                    Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                end
                                                                w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_7)
                                                            end
                                                            A_lvl_2_q += 1
                                                        end
                                                        break
                                                    end
                                                end
                                            end
                                            B_lvl_2_q += 1
                                        else
                                            phase_stop_9 = min(phase_stop_3, B_lvl_2_i)
                                            if B_lvl_2_i == phase_stop_9
                                                B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                                A_lvl_q = (1 - 1) * A_lvl_stop + phase_stop_9
                                                A_lvl_2_q_2 = A_lvl_2_ptr_3[A_lvl_q]
                                                A_lvl_2_q_stop_2 = A_lvl_2_ptr_3[A_lvl_q+1]
                                                if A_lvl_2_q_2 < A_lvl_2_q_stop_2
                                                    A_lvl_2_i1_2 = A_lvl_2_idx_3[A_lvl_2_q_stop_2-1]
                                                else
                                                    A_lvl_2_i1_2 = 0
                                                end
                                                phase_stop_10 = min(A_lvl_2_stop, A_lvl_2_i1_2)
                                                if phase_stop_10 >= 1
                                                    if A_lvl_2_idx_3[A_lvl_2_q_2] < 1
                                                        A_lvl_2_q_2 = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q_2, A_lvl_2_q_stop_2 - 1)
                                                    end
                                                    while true
                                                        A_lvl_2_i_2 = A_lvl_2_idx_3[A_lvl_2_q_2]
                                                        if A_lvl_2_i_2 < phase_stop_10
                                                            A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                            w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i_2
                                                            w_lvl_2_val_3[w_lvl_q_3] = B_lvl_3_val_4 * A_lvl_3_val_5 + w_lvl_2_val_3[w_lvl_q_3]
                                                            if !(w_lvl_tbl_3[w_lvl_q_3])
                                                                w_lvl_tbl_3[w_lvl_q_3] = true
                                                                w_lvl_qos_fill += 1
                                                                if w_lvl_qos_fill > w_lvl_qos_stop
                                                                    w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                    Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                end
                                                                w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i_2)
                                                            end
                                                            A_lvl_2_q_2 += 1
                                                        else
                                                            phase_stop_12 = min(phase_stop_10, A_lvl_2_i_2)
                                                            if A_lvl_2_i_2 == phase_stop_12
                                                                A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                                w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + phase_stop_12
                                                                w_lvl_2_val_3[w_lvl_q_3] += B_lvl_3_val_4 * A_lvl_3_val_5
                                                                if !(w_lvl_tbl_3[w_lvl_q_3])
                                                                    w_lvl_tbl_3[w_lvl_q_3] = true
                                                                    w_lvl_qos_fill += 1
                                                                    if w_lvl_qos_fill > w_lvl_qos_stop
                                                                        w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                        Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                    end
                                                                    w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_12)
                                                                end
                                                                A_lvl_2_q_2 += 1
                                                            end
                                                            break
                                                        end
                                                    end
                                                end
                                                B_lvl_2_q += 1
                                            end
                                            break
                                        end
                                    end
                                end
                                resize!(w_lvl_ptr_3, 1 + 1)
                                resize!(w_lvl_tbl_3, 1A_lvl_2_stop)
                                resize!(w_lvl_srt_3, w_lvl_qos_fill)
                                sort!(w_lvl_srt_3)
                                w_lvl_p_prev = 0
                                for w_lvl_r_2 = 1:w_lvl_qos_fill
                                    w_lvl_p_2 = first(w_lvl_srt_3[w_lvl_r_2])
                                    if w_lvl_p_2 != w_lvl_p_prev
                                        w_lvl_ptr_3[w_lvl_p_prev+1] = w_lvl_r_2
                                        w_lvl_ptr_3[w_lvl_p_2] = w_lvl_r_2
                                    end
                                    w_lvl_p_prev = w_lvl_p_2
                                end
                                w_lvl_ptr_3[w_lvl_p_prev+1] = w_lvl_qos_fill + 1
                                w_lvl_qos_stop = w_lvl_qos_fill
                                resize!(w_lvl_2_val_3, A_lvl_2_stop)
                                w_lvl_r_3 = w_lvl_ptr_3[1]
                                w_lvl_r_stop = w_lvl_ptr_3[1+1]
                                if w_lvl_r_3 != 0 && w_lvl_r_3 < w_lvl_r_stop
                                    w_lvl_i_stop = last(w_lvl_srt_3[w_lvl_r_stop-1])
                                else
                                    w_lvl_i_stop = 0
                                end
                                phase_stop_15 = min(A_lvl_2_stop, w_lvl_i_stop)
                                if phase_stop_15 >= 1
                                    while w_lvl_r_3 + 1 < w_lvl_r_stop && last(w_lvl_srt_3[w_lvl_r_3]) < 1
                                        w_lvl_r_3 += 1
                                    end
                                    while true
                                        w_lvl_i_2 = last(w_lvl_srt_3[w_lvl_r_3])
                                        if w_lvl_i_2 < phase_stop_15
                                            w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                            w_lvl_2_val_4 = w_lvl_2_val_3[w_lvl_q_4]
                                            C_lvl_2_q = (C_lvl_q - 1) * A_lvl_2_stop + w_lvl_i_2
                                            C_lvl_3_val_3[C_lvl_2_q] = w_lvl_2_val_4
                                            w_lvl_r_3 += 1
                                        else
                                            phase_stop_17 = min(phase_stop_15, w_lvl_i_2)
                                            if w_lvl_i_2 == phase_stop_17
                                                w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                                w_lvl_2_val_5 = w_lvl_2_val_3[w_lvl_q_4]
                                                C_lvl_2_q = (C_lvl_q - 1) * A_lvl_2_stop + phase_stop_17
                                                C_lvl_3_val_3[C_lvl_2_q] = w_lvl_2_val_5
                                                w_lvl_r_3 += 1
                                            end
                                            break
                                        end
                                    end
                                end
                            end
                        end
                    end
                end)
                nothing
            end
        end
        resize!(C_lvl_3_val_2, A_lvl_2_stop * B_lvl_stop)
        Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0.0,Float64,Int64}(C_lvl_3_val_2), A_lvl_2_stop), B_lvl_stop))
    end)
end

function spgemm_finch_custom_static_gustavson_dense_helper(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}})
    @inbounds @fastmath(begin
        A_lvl = A.lvl
        A_lvl_stop = A_lvl.shape
        A_lvl_2 = A_lvl.lvl
        A_lvl_2_ptr = A_lvl_2.ptr
        A_lvl_2_idx = A_lvl_2.idx
        A_lvl_2_stop = A_lvl_2.shape
        A_lvl_3 = A_lvl_2.lvl
        A_lvl_3_val = A_lvl_3.val
        B_lvl = B.lvl
        B_lvl_stop = B_lvl.shape
        B_lvl_2 = B_lvl.lvl
        B_lvl_2_ptr = B_lvl_2.ptr
        B_lvl_2_idx = B_lvl_2.idx
        B_lvl_2_stop = B_lvl_2.shape
        B_lvl_3 = B_lvl_2.lvl
        B_lvl_3_val = B_lvl_3.val
        B_lvl_2_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(B_lvl_2_stop) != $(A_lvl_stop))"))

        pos_stop = A_lvl_2_stop * B_lvl_stop
        C_lvl_3_val = zeros(Float64, pos_stop)

        n_threads = Threads.nthreads()
        C_js = [zeros(Float64, A_lvl_2_stop) for _ = 1:n_threads]
        C_idxs = [Int64[] for _ = 1:n_threads]
        C_nfills = [falses(A_lvl_2_stop) for _ = 1:n_threads]

        A_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), A_lvl_2_ptr)
        A_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), A_lvl_2_idx)
        A_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), A_lvl_3_val)
        B_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), B_lvl_2_ptr)
        B_lvl_2_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), B_lvl_2_idx)
        B_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), B_lvl_3_val)
        C_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), C_lvl_3_val)
        C_js_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), C_js)
        C_idxs_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), C_idxs)
        C_nfills_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU(n_threads)), C_nfills)
        Threads.@threads for tid = 1:n_threads
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    A_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), A_lvl_2_ptr_2)
                    A_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), A_lvl_2_idx_2)
                    A_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), A_lvl_3_val_2)
                    B_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), B_lvl_2_ptr_2)
                    B_lvl_2_idx_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), B_lvl_2_idx_2)
                    B_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), B_lvl_3_val_2)
                    C_lvl_3_val_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), C_lvl_3_val_2)
                    C_j = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), C_js_2[tid])
                    C_idx = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), C_idxs_2[tid])
                    C_nfill = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU(n_threads), Finch.Serial()), C_nfills_2[tid])

                    for j in cld(B_lvl_stop, n_threads)*(tid-1)+1:min(B_lvl_stop, cld(B_lvl_stop, n_threads) * tid)
                        for i in C_idx
                            C_nfill[i] = false
                            C_j[i] = 0.0
                        end
                        empty!(C_idx)

                        B_start = B_lvl_2_ptr_3[j]
                        B_stop = B_lvl_2_ptr_3[j+1] - 1
                        for B_ptr in B_start:B_stop
                            k = B_lvl_2_idx_3[B_ptr]
                            B_val = B_lvl_3_val_3[B_ptr]

                            A_start = A_lvl_2_ptr_3[k]
                            A_stop = A_lvl_2_ptr_3[k+1] - 1
                            for A_ptr in A_start:A_stop
                                i = A_lvl_2_idx_3[A_ptr]
                                A_val = A_lvl_3_val_3[A_ptr]
                                C_j[i] += A_val * B_val
                                if !C_nfill[i]
                                    C_nfill[i] = true
                                    push!(C_idx, i)
                                end
                            end
                        end

                        # sort!(C_idx) - only need for SparseList
                        for i in C_idx
                            C_lvl_3_val_3[(j-1)*A_lvl_2_stop+i] = C_j[i]
                        end
                    end
                end)
            end
        end
        Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0.0,Float64,Int64}(C_lvl_3_val), A_lvl_2_stop), B_lvl_stop))
    end)
end

function spgemm_finch_kernel_static_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    _C = Tensor(Dense(Dense(Element(0.0))))
    _w = Tensor(SparseByteMap(Element(0.0)))
    result = @btimed spgemm_finch_kernel_static_gustavson_dense_helper($_A, $_B, $_C, $_w)
    return (; time=result.time,)
end

function spgemm_finch_kernel_greedy_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    _C = Tensor(Dense(Dense(Element(0.0))))
    _w = Tensor(SparseByteMap(Element(0.0)))
    result = @btimed spgemm_finch_kernel_greedy_gustavson_dense_helper($_A, $_B, $_C, $_w)
    return (; time=result.time,)
end

function spgemm_finch_custom_static_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    result = @btimed spgemm_finch_custom_static_gustavson_dense_helper($_A, $_B)
    return (; time=result.time,)
end

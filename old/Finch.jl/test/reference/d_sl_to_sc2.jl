@inbounds begin
        B_lvl = ex.body.body.lhs.tns.tns.lvl
        B_lvl_pos_alloc = length(B_lvl.pos)
        B_lvl_idx_alloc = length(B_lvl.tbl)
        B_lvl_2 = B_lvl.lvl
        B_lvl_2_val_alloc = length(B_lvl.lvl.val)
        B_lvl_2_val = 0.0
        A_lvl = ex.body.body.rhs.tns.tns.lvl
        A_lvl_2 = A_lvl.lvl
        A_lvl_2_pos_alloc = length(A_lvl_2.pos)
        A_lvl_2_idx_alloc = length(A_lvl_2.idx)
        A_lvl_3 = A_lvl_2.lvl
        A_lvl_3_val_alloc = length(A_lvl_2.lvl.val)
        A_lvl_3_val = 0.0
        j_stop = A_lvl_2.I
        i_stop = A_lvl.I
        B_lvl_pos_alloc = length(B_lvl.pos)
        B_lvl_pos_fill = 1
        B_lvl_pos_stop = 2
        B_lvl.pos[1] = 1
        B_lvl.pos[2] = 1
        B_lvl_idx_alloc = length(B_lvl.tbl[1])
        B_lvl_2_val_alloc = (Finch).refill!(B_lvl_2.val, 0.0, 0, 4)
        B_lvl_pos_alloc < 1 + 1 && (B_lvl_pos_alloc = (Finch).regrow!(B_lvl.pos, B_lvl_pos_alloc, 1 + 1))
        B_lvl_pos_stop = 1 + 1
        B_lvl_q = B_lvl.pos[B_lvl_pos_fill]
        for B_lvl_p_2 = B_lvl_pos_fill:1
            B_lvl.pos[B_lvl_p_2] = B_lvl_q
        end
        for i = 1:i_stop
            A_lvl_q = (1 - 1) * A_lvl.I + i
            A_lvl_2_q = A_lvl_2.pos[A_lvl_q]
            A_lvl_2_q_stop = A_lvl_2.pos[A_lvl_q + 1]
            if A_lvl_2_q < A_lvl_2_q_stop
                A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
                A_lvl_2_i1 = A_lvl_2.idx[A_lvl_2_q_stop - 1]
            else
                A_lvl_2_i = 1
                A_lvl_2_i1 = 0
            end
            j = 1
            j_start = j
            phase_start = j_start
            phase_stop = (min)(A_lvl_2_i1, j_stop)
            if phase_stop >= phase_start
                j = j
                j = phase_start
                while A_lvl_2_q + 1 < A_lvl_2_q_stop && A_lvl_2.idx[A_lvl_2_q] < phase_start
                    A_lvl_2_q += 1
                end
                while j <= phase_stop
                    j_start_2 = j
                    A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
                    phase_stop_2 = (min)(A_lvl_2_i, phase_stop)
                    j_2 = j
                    if A_lvl_2_i == phase_stop_2
                        A_lvl_3_val = A_lvl_3.val[A_lvl_2_q]
                        j_3 = phase_stop_2
                        B_lvl_guard = true
                        B_lvl_2_val_alloc < B_lvl_q && (B_lvl_2_val_alloc = (Finch).refill!(B_lvl_2.val, 0.0, B_lvl_2_val_alloc, B_lvl_q))
                        B_lvl_2_val = B_lvl_2.val[B_lvl_q]
                        B_lvl_guard = false
                        B_lvl_guard = false
                        B_lvl_2_val = (+)(A_lvl_3_val, B_lvl_2_val)
                        B_lvl_2.val[B_lvl_q] = B_lvl_2_val
                        if !B_lvl_guard
                            if B_lvl_idx_alloc < B_lvl_q
                                (Finch).regrow!(B_lvl.tbl[1], B_lvl_idx_alloc, B_lvl_q)
                                B_lvl_idx_alloc = (Finch).regrow!(B_lvl.tbl[2], B_lvl_idx_alloc, B_lvl_q)
                            end
                            (B_lvl.tbl[1])[B_lvl_q] = i
                            (B_lvl.tbl[2])[B_lvl_q] = j_3
                            B_lvl_q += 1
                        end
                        A_lvl_2_q += 1
                    else
                    end
                    j = phase_stop_2 + 1
                end
                j = phase_stop + 1
            end
            j_start = j
            phase_start_3 = j_start
            phase_stop_3 = j_stop
            if phase_stop_3 >= phase_start_3
                j_4 = j
                j = phase_stop_3 + 1
            end
        end
        B_lvl.pos[1 + 1] = B_lvl_q
        B_lvl_pos_fill = 1 + 1
        q = B_lvl.pos[B_lvl_pos_fill]
        for p = B_lvl_pos_fill:B_lvl_pos_stop
            B_lvl.pos[p] = q
        end
        B_lvl_pos_alloc = 1 + 1
        resize!(B_lvl.pos, B_lvl_pos_alloc)
        B_lvl_idx_alloc = B_lvl.pos[B_lvl_pos_alloc] - 1
        for idx = B_lvl.tbl
            resize!(idx, B_lvl_idx_alloc)
        end
        resize!(B_lvl_2.val, B_lvl_idx_alloc)
        (B = Fiber((Finch.SparseCooLevel){2, Tuple{Int64, Int64}, Int64}((A_lvl.I, A_lvl_2.I), B_lvl.tbl, B_lvl.pos, B_lvl_2), (Finch.Environment)(; )),)
    end

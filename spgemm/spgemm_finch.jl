using Finch
using BenchmarkTools

function spgemm_finch_custom_gustavson_dense_helper(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}})
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

function spgemm_finch_custom_gustavson_dense(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    result = @btimed spgemm_finch_custom_gustavson_dense_helper($_A, $_B)
    return (; time=result.time, C=result.value)
end


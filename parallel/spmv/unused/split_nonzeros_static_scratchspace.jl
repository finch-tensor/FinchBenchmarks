using Finch
using BenchmarkTools
using Base.Threads

function split_nonzeros_static_scratchspace_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                split_nonzeros_static_scratchspace(_y, _A, _x)
        end
        return (; time=time, y=_y)
end

function split_nonzeros_static_scratchspace(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
        @inbounds @fastmath(begin
                y_lvl = y.lvl # DenseLevel
                # y_lvl_2 = y_lvl.lvl # ElementLevel
                y_lvl_val = y_lvl.lvl.val # Vector{Float64}

                A_lvl = A.lvl # DenseLevel
                A_lvl_2 = A_lvl.lvl # SparseListLevel
                A_lvl_ptr = A_lvl_2.ptr # Vector{Int64}
                A_lvl_idx = A_lvl_2.idx # Vector{Int64}
                # A_lvl_3 = A_lvl_2.lvl # ElementLevel
                A_lvl_2_val = A_lvl_2.lvl.val # Vector{Float64}

                x_lvl = x.lvl # DenseLevel
                # x_lvl_2 = x_lvl.lvl # ElementLevel
                x_lvl_val = x_lvl.lvl.val # Vector{Float64}

                x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
                Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
                Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)

                num_threads = Threads.nthreads()
                y_temps = [zeros(Float64, y_lvl.shape) for _ in 1:num_threads]

                # Load Balancing
                num_nz = A_lvl_ptr[A_lvl.shape+1] - 1
                start_indices = [1 + div(k * num_nz, num_threads) for k in 0:num_threads]
                start_cols = zeros(Int64, num_threads + 1)

                col = 2
                target_pos = 1
                target_index = start_indices[target_pos]
                while (col <= A_lvl.shape + 1 && target_pos <= num_threads)
                        if (A_lvl_ptr[col] > target_index)
                                start_cols[target_pos] = col - 1
                                target_pos += 1
                                target_index = start_indices[target_pos]
                        else
                                col += 1
                        end
                end
                start_cols[num_threads+1] = A_lvl.shape

                Threads.@threads for k = 1:num_threads
                        start_index = start_indices[k]
                        end_index = start_indices[k+1] - 1
                        for j = start_cols[k]:start_cols[k+1]
                                for q in max(A_lvl_ptr[j], start_index):min(A_lvl_ptr[j+1] - 1, end_index)
                                        i = A_lvl_idx[q]
                                        temp_val = A_lvl_2_val[q] * x_lvl_val[j]
                                        y_temps[k][i] += temp_val
                                end
                        end
                end

                Threads.@threads for k = 1:num_threads
                        for j = 1:num_threads
                                for i = 1+div((k - 1) * y_lvl.shape, num_threads):div(k * y_lvl.shape, num_threads)
                                        y_lvl_val[i] += y_temps[j][i]
                                end
                        end
                end
        end)
end

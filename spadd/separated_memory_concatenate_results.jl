using Finch
using BenchmarkTools
using Base.Threads

include("concat.jl")

function separated_memory_concatenate_results_add(A, B, num_cpu)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)

    time = @belapsed kernel($(_A), $(_B))

    C = kernel(_A, _B)

    return (; time=time, C=C)
end

function kernel(_A, _B)
    num_threads = Threads.nthreads()
    partial_sum = Vector{Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}(undef, num_threads)
    partial_nonzero_ptr = Vector{Int64}(undef, num_threads + 1)
    partial_nonzero_ptr[1] = 0
    partial_column = Vector{Int64}(undef, num_threads + 1)
    partial_column[1] = 0

    num_col = size(_A)[2]
    Threads.@threads for k = 1:num_threads
        start_col = 1 + div((k - 1) * num_col, num_threads)
        stop_col = div(k * num_col, num_threads)
        partial_column[k+1] = stop_col
        result = partial_add(_A, _B, start_col, stop_col)
        partial_sum[k] = result.tensor
        partial_nonzero_ptr[k+1] = result.num_nonzero
    end

    for i in 2:length(partial_nonzero_ptr)
        partial_nonzero_ptr[i] += partial_nonzero_ptr[i-1]
    end

    return concat_vec(partial_sum, partial_nonzero_ptr, partial_column)
end

# Add A and B from column start_col to stop_col (inclusive)
function partial_add(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, start_col::Int64, stop_col::Int64)
    @inbounds @fastmath(begin
        A_lvl = A.lvl # DenseLevel
        A_lvl_2 = A_lvl.lvl # SparseListLevel
        A_lvl_ptr = A_lvl_2.ptr # Vector{Int64}
        A_lvl_idx = A_lvl_2.idx # Vector{Int64}
        # A_lvl_3 = A_lvl_2.lvl # ElementLevel
        A_lvl_2_val = A_lvl_2.lvl.val # Vector{Float64}

        B_lvl = B.lvl # DenseLevel
        B_lvl_2 = B_lvl.lvl # SparseListLevel
        B_lvl_ptr = B_lvl_2.ptr # Vector{Int64}
        B_lvl_idx = B_lvl_2.idx # Vector{Int64}
        # B_lvl_3 = B_lvl_2.lvl # ElementLevel
        B_lvl_2_val = B_lvl_2.lvl.val # Vector{Float64}

        # Assertion
        # A_lvl.shape == B_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(B_lvl.shape))"))
        # A_lvl_2.shape == B_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl_2.shape) != $(B_lvl_2.shape))"))
        # A_lvl.shape >= stop_col || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) < $(stop_col))"))

        C_lvl_2_val = Vector{Float64}()
        C_lvl_idx = Vector{Int64}()
        C_lvl_ptr = Vector{Int64}([1])
        current_ptr = 1
        for j = start_col:stop_col
            A_idx = A_lvl_ptr[j]
            B_idx = B_lvl_ptr[j]
            while A_idx < A_lvl_ptr[j+1] && B_idx < B_lvl_ptr[j+1]
                current_ptr += 1
                A_row_idx = A_lvl_idx[A_idx]
                B_row_idx = B_lvl_idx[B_idx]

                if A_row_idx < B_row_idx
                    push!(C_lvl_2_val, A_lvl_2_val[A_idx])
                    push!(C_lvl_idx, A_row_idx)
                    A_idx += 1
                elseif A_row_idx > B_row_idx
                    push!(C_lvl_2_val, B_lvl_2_val[B_idx])
                    push!(C_lvl_idx, B_row_idx)
                    B_idx += 1
                else
                    push!(C_lvl_2_val, A_lvl_2_val[A_idx] + B_lvl_2_val[B_idx])
                    push!(C_lvl_idx, A_row_idx)
                    A_idx += 1
                    B_idx += 1
                end
            end

            append!(C_lvl_2_val, @view A_lvl_2_val[A_idx:A_lvl_ptr[j+1]-1])
            append!(C_lvl_idx, @view A_lvl_idx[A_idx:A_lvl_ptr[j+1]-1])
            current_ptr += A_lvl_ptr[j+1] - A_idx

            append!(C_lvl_2_val, @view B_lvl_2_val[B_idx:B_lvl_ptr[j+1]-1])
            append!(C_lvl_idx, @view B_lvl_idx[B_idx:B_lvl_ptr[j+1]-1])
            current_ptr += B_lvl_ptr[j+1] - B_idx

            append!(C_lvl_ptr, current_ptr)
        end

        C_lvl_3 = Element{0.0,Float64,Int64}(C_lvl_2_val)
        C_lvl_shape = A_lvl_2.shape

        C_lvl_2 = SparseList{Int64}(C_lvl_3, C_lvl_shape, C_lvl_ptr, C_lvl_idx)
        C_lvl = Dense{Int64}(C_lvl_2, stop_col - start_col + 1)

        C = Tensor(C_lvl)
        return (tensor=C, num_nonzero=current_ptr - 1)
    end)
end

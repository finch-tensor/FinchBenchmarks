using Finch
using BenchmarkTools
using Graphs
using Metis
using SparseArrays

function create_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})
    tns_lvl = A.body.lvl
    tns_lvl_2 = tns_lvl.lvl
    tns_lvl_ptr = tns_lvl_2.ptr
    tns_lvl_idx = tns_lvl_2.idx

    tns_lvl.shape == tns_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(tns_lvl.shape) != $(tns_lvl_2.shape))"))

    graph = SimpleGraph(tns_lvl.shape)
    for v in 1:tns_lvl.shape
        for ptr in tns_lvl_ptr[v]:tns_lvl_ptr[v+1]-1
            add_edge!(graph, v, tns_lvl_idx[ptr])
        end
    end

    # Partition the graph
    positions = Metis.partition(graph, Threads.nthreads(); alg=:KWAY)

    # create permutation for the graph 
    perm = sortperm(positions)
    return perm
end

# permute the vector v
function vector_permutation(v::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, perm::Vector{Int64})
    v_lvl = v.lvl
    v_lvl_val = v_lvl.lvl.val
    v_perm = v_lvl_val[perm]
    return Tensor(Dense(Element(0.0)), v_perm)
end

# permute columns of matrix A
function matrix_col_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, perm::Vector{Int64})
    tns_lvl = A.body.lvl
    tns_lvl_2 = tns_lvl.lvl
    tns_lvl_ptr = tns_lvl_2.ptr
    tns_lvl_idx = tns_lvl_2.idx
    tns_lvl_2_val = tns_lvl_2.lvl.val

    _A = SparseMatrixCSC(tns_lvl.shape, tns_lvl_2.shape, tns_lvl_ptr, tns_lvl_idx, tns_lvl_2_val)
    A_perm = _A[:, perm]
    return Tensor(Dense(SparseList(Element(0.0))), A_perm)
end

"""
    merge_path_search(diagonal::Int64, num_rows::Int64, num_nzs::Int64, row_ptr::Vector{Int64})

Find the intersecting coordinate between the merge path and the diagonal

# Arguments
- `diagonal::Int64`: the line s.t. row_idx + nz_idx = diagonal (diagonal >= 2)
- `num_rows::Int64`: the number of rows
- `num_nzs::Int64`: the number of nonzeros
- `row_ptr::Vector{Int64}`: the row ptr representing a cumulative number of nonzero elements
"""
function merge_path_search(diagonal::Int64, num_rows::Int64, num_nzs::Int64, row_ptr::Vector{Int64})
    x_min = max(diagonal - num_nzs - 1, 1)
    x_max = min(diagonal - 1, num_rows + 1)

    while x_min < x_max
        pivot = (x_min + x_max) >> 1
        if row_ptr[pivot+1] <= diagonal - pivot - 1
            x_min = pivot + 1
        else
            x_max = pivot
        end
    end

    return (min(x_min, num_rows + 1), diagonal - x_min)
end

function graph_partition_merge_helper(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
    @inbounds @fastmath(begin
        y_lvl = y.lvl
        y_lvl_val = y_lvl.lvl.val
        A_lvl = A.lvl
        A_lvl_2 = A_lvl.lvl
        A_lvl_ptr = A_lvl_2.ptr
        A_lvl_idx = A_lvl_2.idx
        A_lvl_2_val = A_lvl_2.lvl.val
        x_lvl = x.lvl
        x_lvl_val = x_lvl.lvl.val
        x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
        Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
        Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
        val = y_lvl_val

        y_lvl_val = (Finch).moveto(y_lvl_val, CPU(Threads.nthreads()))
        x_lvl_val = (Finch).moveto(x_lvl_val, CPU(Threads.nthreads()))
        A_lvl_ptr = (Finch).moveto(A_lvl_ptr, CPU(Threads.nthreads()))
        A_lvl_idx = (Finch).moveto(A_lvl_idx, CPU(Threads.nthreads()))
        A_lvl_2_val = (Finch).moveto(A_lvl_2_val, CPU(Threads.nthreads()))

        # Custom Variables
        num_rows = A_lvl.shape
        num_nzs = last(A_lvl_ptr) - 1
        num_merge_items = num_rows + num_nzs # number of rows + number of nonzeros
        items_per_thread = fld(num_merge_items + Threads.nthreads() - 1, Threads.nthreads())
        row_carry_out = Vector{Int64}(undef, Threads.nthreads())
        value_carry_out = Vector{Float64}(undef, Threads.nthreads())

        row_carry_out = (Finch).moveto(row_carry_out, CPU(Threads.nthreads()))
        value_carry_out = (Finch).moveto(value_carry_out, CPU(Threads.nthreads()))

        Threads.@threads for i_4 = 1:Threads.nthreads()
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    diagonal = min(items_per_thread * (i_4 - 1) + 2, num_merge_items + 2)
                    diagonal_end = min(diagonal + items_per_thread, num_merge_items + 2)
                    x_coord, y_coord = merge_path_search(diagonal, num_rows, num_nzs, A_lvl_ptr)
                    x_coord_end, y_coord_end = merge_path_search(diagonal_end, num_rows, num_nzs, A_lvl_ptr)

                    running_total = 0.0
                    while x_coord < x_coord_end
                        while y_coord < A_lvl_ptr[x_coord+1]
                            running_total += A_lvl_2_val[y_coord] * x_lvl_val[A_lvl_idx[y_coord]]
                            y_coord += 1
                        end
                        val[x_coord] = running_total
                        running_total = 0.0
                        x_coord += 1
                    end

                    while y_coord < y_coord_end
                        running_total += A_lvl_2_val[y_coord] * x_lvl_val[A_lvl_idx[y_coord]]
                        y_coord += 1
                    end

                    row_carry_out[i_4] = x_coord_end
                    value_carry_out[i_4] = running_total
                end)
            end
        end

        for i = 1:Threads.nthreads()
            if row_carry_out[i] < num_rows + 1
                val[row_carry_out[i]] += value_carry_out[i]
            end
        end
        resize!(val, A_lvl.shape)
    end)
end

function graph_partition_merge(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)

    perm = create_permutation(_A)
    _A = matrix_col_permutation(_A, perm)
    time = @belapsed graph_partition_merge_helper($_y, $_A, $_x)
    _y = vector_permutation(_y, invperm(perm))
    return (; time=time, y=_y)
end


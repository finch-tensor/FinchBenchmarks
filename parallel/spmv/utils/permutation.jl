using Metis: idx_t
using Metis
using SparseArrays
using Finch

"""
    create_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})

Return a permutation of a matrix that will minimize communication of x value in Ax if we group the matrix in to num cores groups of equal size

# Arguments
- `A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}`: a matrix to create permutation on, the matrix must be NxN (rows == columns)
"""
function create_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})
    tns_lvl = A.body.lvl
    tns_lvl_2 = tns_lvl.lvl
    tns_lvl_ptr = tns_lvl_2.ptr
    tns_lvl_idx = tns_lvl_2.idx

    tns_lvl.shape == tns_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(tns_lvl.shape) != $(tns_lvl_2.shape))"))

    nvtxs = convert(idx_t, tns_lvl.shape)
    adjncy_temp = [idx_t[] for _ in 1:tns_lvl.shape]

    for v in 1:tns_lvl.shape
        for ptr in tns_lvl_ptr[v]:tns_lvl_ptr[v+1]-1
            push!(adjncy_temp[tns_lvl_idx[ptr]], v)
        end
        append!(adjncy_temp[v], tns_lvl_idx[tns_lvl_ptr[v]:tns_lvl_ptr[v+1]-1])
    end

    xadj = idx_t[]
    push!(xadj, convert(idx_t, 1))
    for v in 1:tns_lvl.shape
        push!(xadj, xadj[v] + length(adjncy_temp[v]))
    end

    adjncy = vcat(adjncy_temp...)

    graph = Metis.Graph(nvtxs, xadj, adjncy)

    # Partition the graph
    positions = Metis.partition(graph, Threads.nthreads(); alg=:KWAY)

    # create permutation for the graph 
    perm = sortperm(positions)
    return perm
end

"""
    create_weighted_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})

Return a permutation of a matrix that will minimize communication of x value in Ax if we group the matrix in to num cores groups of equal size, where size for each row equals the number of nnz in that row + 1

# Arguments
- `A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}`: a matrix to create permutation on, the matrix must be NxN (rows == columns)
"""
function create_weighted_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})
    tns_lvl = A.body.lvl
    tns_lvl_2 = tns_lvl.lvl
    tns_lvl_ptr = tns_lvl_2.ptr
    tns_lvl_idx = tns_lvl_2.idx

    tns_lvl.shape == tns_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(tns_lvl.shape) != $(tns_lvl_2.shape))"))

    nvtxs = convert(idx_t, tns_lvl.shape)
    adjncy_temp = [idx_t[] for _ in 1:tns_lvl.shape]
    vwgt = zeros(idx_t, tns_lvl.shape)

    for v in 1:tns_lvl.shape
        for ptr in tns_lvl_ptr[v]:tns_lvl_ptr[v+1]-1
            push!(adjncy_temp[tns_lvl_idx[ptr]], v)
        end
        vwgt[v] = tns_lvl_ptr[v+1] - tns_lvl_ptr[v] + 1 # the last 1 is for the row
        append!(adjncy_temp[v], tns_lvl_idx[tns_lvl_ptr[v]:tns_lvl_ptr[v+1]-1])
    end

    xadj = idx_t[]
    push!(xadj, convert(idx_t, 1))
    for v in 1:tns_lvl.shape
        push!(xadj, xadj[v] + length(adjncy_temp[v]))
    end

    adjncy = vcat(adjncy_temp...)

    graph = Metis.Graph(nvtxs, xadj, adjncy, vwgt)

    # Partition the graph
    positions = Metis.partition(graph, Threads.nthreads(); alg=:KWAY)

    # create permutation for the graph 
    perm = sortperm(positions)
    return perm
end

"""
    vector_permutation(v::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, perm::Vector{Int64})

Create a permutation of a vector

# Arguments
- `v::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}`: vector to be permuted
- `perm::Vector{Int64}`: permutation vector, must have size at most length(v)
"""
function vector_permutation(v::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, perm::Vector{Int64})
    v_lvl = v.lvl
    v_lvl_val = v_lvl.lvl.val
    v_perm = v_lvl_val[perm]
    return Tensor(Dense(Element(0.0)), v_perm)
end

"""
    matrix_col_permutation(A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, perm::Vector{Int64})

Create a column permutation of a matrix

# Arguments
- `A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}`: matrix to be permuted
- `perm::Vector{Int64}`: permutation vector, must have size at most number of columns of A
"""
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

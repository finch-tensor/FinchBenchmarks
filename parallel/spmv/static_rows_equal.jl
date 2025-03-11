using Finch
using BenchmarkTools

function static_rows_equal_helper(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
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

        Threads.@threads for t = 1:Threads.nthreads()
            Finch.@barrier begin
                @inbounds @fastmath(begin
                    for i = 1+div((t - 1) * tns_lvl.shape, Threads.nthreads()):div(t * tns_lvl.shape, Threads.nthreads())
                        for ptr = tns_lvl_ptr[i]:tns_lvl_ptr[i+1]-1
                            val[i] += tns_lvl_2_val[ptr] * x_lvl_val[tns_lvl_idx[ptr]]
                        end
                    end
                end)
            end
        end

        resize!(val, tns_lvl.shape)
    end)
end

function static_rows_equal(y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)

    time = @belapsed static_rows_equal_helper($_y, $_A, $_x)
    return (; time=time, y=_y)
end

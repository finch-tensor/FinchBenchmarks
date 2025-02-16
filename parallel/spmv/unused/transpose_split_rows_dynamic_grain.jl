using Finch
using BenchmarkTools
using Base.Threads


function transpose_split_rows_dynamic_grain_mul(grain_size)
    return (y, A, x) -> transpose_split_rows_dynamic_grain_helper(grain_size, y, A, x)
end

function transpose_split_rows_dynamic_grain_helper(grain_size, y, A, x)
    _y = Tensor(Dense(Element(0.0)), y)
    _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
    _x = Tensor(Dense(Element(0.0)), x)
    time = @belapsed begin
        (grain_size, _y, _A, _x) = $(grain_size, _y, _A, _x)
        transpose_split_rows_dynamic_grain(grain_size, _y, _A, _x)
    end
    return (; time=time, y=_y)
end

function transpose_split_rows_dynamic_grain(grain_size::Int64, y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
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

        cap_size = div(tns_lvl.shape, grain_size) * grain_size

        Threads.@threads for group = 1:grain_size:cap_size
            Finch.@barrier group grain_size tns_lvl_ptr tns_lvl_idx y_lvl_val tns_lvl_2_val x_lvl_val begin
                for i = group:group+grain_size-1
                    for q in tns_lvl_ptr[i]:tns_lvl_ptr[i+1]-1
                        j = tns_lvl_idx[q]
                        y_lvl_val[i] += tns_lvl_2_val[q] * x_lvl_val[j]
                    end
                end
            end
        end

        Threads.@threads for i = cap_size+1:tns_lvl.shape
            Finch.@barrier tns_lvl_ptr tns_lvl_idx y_lvl_val tns_lvl_2_val x_lvl_val i begin
                for q in tns_lvl_ptr[i]:tns_lvl_ptr[i+1]-1
                    j = tns_lvl_idx[q]
                    y_lvl_val[i] += tns_lvl_2_val[q] * x_lvl_val[j]
                end
            end
        end
    end)
end


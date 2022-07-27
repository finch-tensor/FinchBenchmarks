@kwdef mutable struct VirtualAbstractArray
    ndims
    name
    ex
end

function getdims(arr::VirtualAbstractArray, ctx::LowerJulia, mode) where {T <: AbstractArray}
    dims = map(i -> Symbol(arr.name, :_mode, i, :_stop), 1:arr.ndims)
    push!(ctx.preamble, quote
        ($(dims...),) = size($(arr.ex))
    end)
    return map(i->Extent(1, Virtual{Int}(dims[i])), 1:arr.ndims)
end

getname(arr::VirtualAbstractArray) = arr.name
setname(arr::VirtualAbstractArray, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)

function (ctx::LowerJulia)(arr::VirtualAbstractArray, ::DefaultStyle)
    return arr.ex
end

function virtualize(ex, ::Type{<:AbstractArray{T, N}}, ctx, tag=:tns) where {T, N}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualAbstractArray(N, tag, sym)
end

function initialize!(arr::VirtualAbstractArray, ctx::LowerJulia, mode::Union{Write, Update}, idxs...)
    push!(ctx.preamble, quote
        fill!($(arr.ex), 0) #TODO
    end)
    access(arr, mode, idxs...)
end

isliteral(::VirtualAbstractArray) = false

default(::VirtualAbstractArray) = 0
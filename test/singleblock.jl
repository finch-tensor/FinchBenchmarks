mutable struct SingleBlock{D, Tv, Ti} <: AbstractVector{Tv}
    I::Ti
    start::Ti
    stop::Ti
    val::Vector{Tv}
end

function SingleBlock{D}(I::Ti, start::Ti, stop::Ti, val::Vector{Tv}) where {D, Ti, Tv}
    SingleBlock{D, Tv, Ti}(I, start, stop, val)
end

Base.size(vec::SingleBlock) = (vec.I,)

function Base.getindex(vec::SingleBlock{D, Tv, Ti}, i) where {D, Tv, Ti}
    if i < vec.start
        D
    elseif i <= vec.stop
        vec.val[i - vec.start + 1]
    else
        D
    end
end

mutable struct VirtualSingleBlock{Tv, Ti}
    ex
    name
    D
end

function Finch.virtualize(ex, ::Type{SingleBlock{D, Tv, Ti}}, ctx, tag=:tns) where {D, Tv, Ti}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSingleBlock{Tv, Ti}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSingleBlock) = tns.ex

function Finch.initialize!(arr::VirtualSingleBlock{D, Tv}, ctx::Finch.LowerJulia, mode::Union{Write, Update}, idxs...) where {D, Tv}
    push!(ctx.preamble, quote
        fill!($(arr.ex).val, D)
    end)
    access(arr, mode, idxs...)
end 

function Finch.getdims(arr::VirtualSingleBlock{Tv, Ti}, ctx::Finch.LowerJulia, mode) where {Tv, Ti}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(1, Virtual{Ti}(ex)),)
end
Finch.setdims!(arr::VirtualSingleBlock{Tv, Ti}, ctx::Finch.LowerJulia, mode, dims...) where {Tv, Ti} = arr
Finch.getname(arr::VirtualSingleBlock) = arr.name
Finch.setname(arr::VirtualSingleBlock, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function (ctx::Finch.Stylize{LowerJulia})(node::Access{<:VirtualSingleBlock})
    if ctx.root isa Loop && ctx.root.idx == get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        mapreduce(ctx, result_style, arguments(node))
    end
end

function (ctx::Finch.ChunkifyVisitor)(node::Access{VirtualSingleBlock{Tv, Ti}, Read}, ::Finch.DefaultStyle) where {Tv, Ti}
    vec = node.tns
    if getname(ctx.idx) == getname(node.idxs[1])
        tns = Pipeline([
            Phase(
                stride = (ctx, idx, ext) -> :($(vec.ex).start - 1),
                body = (start, step) -> Run(body = Simplify(vec.D))
            ),
            Phase(
                stride = (ctx, idx, ext) -> :($(vec.ex).stop),
                body = (start, step) -> Lookup(
                    body = (i) -> :($(vec.ex).val[$(ctx.ctx(i)) - $(vec.ex).start + 1]) #TODO all of these functions should really have a ctx
                )
            ),
            Phase(body = (start, step) -> Run(body = Simplify(vec.D)))
        ])
        Access(tns, node.mode, node.idxs)
    else
        node
    end
end

Finch.register()
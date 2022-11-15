mutable struct SingleSpike{D, Tv} <: AbstractVector{Tv}
    n
    tail::Tv
end

Base.size(vec::SingleSpike) = (vec.n,)

function SingleSpike{D}(n, tail::Tv) where {D, Tv}
    SingleSpike{D, Tv}(n, tail)
end

function Base.getindex(vec::SingleSpike{D}, i) where {D}
    i == vec.n ? vec.tail : D
end

mutable struct VirtualSingleSpike{Tv}
    ex
    name
    D
end

Finch.IndexNotation.isliteral(::VirtualSingleSpike) = false

function Finch.virtualize(ex, ::Type{SingleSpike{D, Tv}}, ctx, tag=:tns) where {D, Tv}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSingleSpike{Tv}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSingleSpike) = tns.ex

function Finch.getsize(arr::VirtualSingleSpike{Tv}, ctx::Finch.LowerJulia, mode) where {Tv}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(literal(1), value(ex, Int)),)
end
Finch.setsize!(arr::VirtualSingleSpike, ctx::Finch.LowerJulia, mode, dims...) = arr
Finch.getname(arr::VirtualSingleSpike) = arr.name
Finch.setname(arr::VirtualSingleSpike, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function Finch.stylize_access(node, ctx::Finch.Stylize{LowerJulia}, ::VirtualSingleSpike)
    if ctx.root isa IndexNode && ctx.root.kind === loop && ctx.root.idx == get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        Finch.DefaultStyle()
    end
end

function Finch.chunkify_access(node, ctx, vec::VirtualSingleSpike{Tv}) where {Tv}
    if getname(ctx.idx) == getname(node.idxs[1])
        tns = Spike(
            body = Simplify(zero(Tv)),
            tail = value(:($(vec.ex).tail), Tv)
        )
        access(tns, node.mode, node.idxs...)
    else
        node
    end
end

Finch.register()
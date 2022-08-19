mutable struct SimpleJumpVector{D, Tv, Ti} <: AbstractVector{Tv}
    idx::Vector{Ti}
    val::Vector{Tv}
end

function SimpleJumpVector{D}(idx::Vector{Ti}, val::Vector{Tv}) where {D, Ti, Tv}
    SimpleJumpVector{D, Tv, Ti}(idx, val)
end

Base.size(vec::SimpleJumpVector) = (vec.idx[end] - 1,)

function Base.getindex(vec::SimpleJumpVector{D, Tv, Ti}, i) where {D, Tv, Ti}
    p = findfirst(j->j >= i, vec.idx)
    vec.idx[p] == i ? vec.val[p] : D
end

mutable struct VirtualSimpleJumpVector{Tv, Ti}
    ex
    name
    D
end

function Finch.virtualize(ex, ::Type{SimpleJumpVector{D, Tv, Ti}}, ctx, tag=:tns) where {D, Tv, Ti}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSimpleJumpVector{Tv, Ti}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSimpleJumpVector) = tns.ex

function Finch.initialize!(arr::VirtualSimpleJumpVector{D, Tv}, ctx::Finch.LowerJulia, mode::Union{Write, Update}, idxs...) where {D, Tv}
    push!(ctx.preamble, quote
        $(arr.ex).idx = [$(arr.ex).idx[end]]
        $(arr.ex).val = $Tv[]
    end)
    access(arr, mode, idxs...)
end 

function Finch.getdims(arr::VirtualSimpleJumpVector{Tv, Ti}, ctx::Finch.LowerJulia, mode) where {Tv, Ti}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(1, Virtual{Ti}(ex)),)
end
Finch.setdims!(arr::VirtualSimpleJumpVector{Tv, Ti}, ctx::Finch.LowerJulia, mode, dims...) where {Tv, Ti} = arr
Finch.getname(arr::VirtualSimpleJumpVector) = arr.name
Finch.setname(arr::VirtualSimpleJumpVector, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function (ctx::Finch.Stylize{LowerJulia})(node::Access{<:VirtualSimpleJumpVector})
    if ctx.root isa Loop && ctx.root.idx == get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        mapreduce(ctx, result_style, arguments(node))
    end
end

function (ctx::Finch.ChunkifyVisitor)(node::Access{VirtualSimpleJumpVector{Tv, Ti}, Read}, ::Finch.DefaultStyle) where {Tv, Ti}
    vec = node.tns
    my_i = ctx.ctx.freshen(getname(vec), :_i0)
    my_i′ = ctx.ctx.freshen(getname(vec), :_i1)
    my_p = ctx.ctx.freshen(getname(vec), :_p)
    if getname(ctx.idx) == getname(node.idxs[1])
        tns = Thunk(
            preamble = quote
                $my_p = 1
                $my_i = 1
                $my_i′ = $(vec.ex).idx[$my_p]
            end,
            body = Jumper(
                body = Jump(
                    seek = (ctx, ext) -> quote
                        $my_p = searchsortedfirst($(vec.ex).idx, $(ctx(getstart(ext))), $my_p, length($(vec.ex).idx), Base.Forward)
                        $my_i = $(ctx(getstart(ext)))
                        $my_i′ = $(vec.ex).idx[$my_p]
                    end,
                    stride = (ctx, ext) -> my_i′,
                    body = (ctx, ext, ext_2) -> begin
                        Switch([
                            :($(ctx(getstop(ext_2))) == $my_i′) => Thunk(
                                body = Spike(
                                    body = Simplify(zero(Tv)),
                                    tail = Virtual{Tv}(:($(vec.ex).val[$my_p])),
                                ),
                                epilogue = quote
                                    $my_p += 1
                                    $my_i = $my_i′ + 1
                                    $my_i′ = $(vec.ex).idx[$my_p]
                                end
                            ),
                            true => Stepper(
                                seek = (ctx, ext) -> quote
                                    $my_p = searchsortedfirst($(vec.ex).idx, $(ctx(getstart(ext))), $my_p, length($(vec.ex).idx), Base.Forward)
                                    $my_i = $(ctx(getstart(ext)))
                                    $my_i′ = $(vec.ex).idx[$my_p]
                                end,
                                body = Step(
                                    stride = (ctx, idx, ext) -> my_i′,
                                    chunk = Spike(
                                        body = Simplify(zero(Tv)),
                                        tail = Virtual{Tv}(:($(vec.ex).val[$my_p])),
                                    ),
                                    next = (ctx, idx, ext) -> quote
                                        $my_p += 1
                                        $my_i = $my_i′ + 1
                                        $my_i′ = $(vec.ex).idx[$my_p]
                                    end
                                )
                            )
                        ])
                    end
                )
            )
        )
        Access(tns, node.mode, node.idxs)
    else
        node
    end
end

function (ctx::Finch.ChunkifyVisitor)(node::Access{VirtualSimpleJumpVector{Tv, Ti}, <:Union{Write, Update}}, ::Finch.DefaultStyle) where {Tv, Ti}
    vec = node.tns
    my_p = ctx.ctx.freshen(node.tns.name, :_p)
    my_I = ctx.ctx.freshen(node.tns.name, :_I)
    if getname(ctx.idx) == getname(node.idxs[1])
        push!(ctx.ctx.preamble, quote
            $my_p = 0
            $my_I = $(ctx.ctx(getstop(ctx.ctx.dims[getname(node.idxs[1])]))) + 1 #TODO is this okay? Should chunkify tell us which extent to use?
            $(vec.ex).idx = $Ti[$my_I]
            $(vec.ex).val = $Tv[]
        end)
        tns = AcceptSpike(
            val = vec.D,
            tail = (ctx, idx) -> Thunk(
                preamble = quote
                    push!($(vec.ex).idx, $my_I)
                    push!($(vec.ex).val, zero($Tv))
                    $my_p += 1
                end,
                body = Virtual{Tv}(:($(vec.ex).val[$my_p])),
                epilogue = quote
                    $(vec.ex).idx[$my_p] = $(ctx(idx))
                end
            )
        )
        Access(tns, node.mode, node.idxs)
    else
        node
    end
end

Finch.register()
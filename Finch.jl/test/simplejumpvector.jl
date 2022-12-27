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

Finch.IndexNotation.isliteral(::VirtualSimpleJumpVector) = false

function Finch.virtualize(ex, ::Type{SimpleJumpVector{D, Tv, Ti}}, ctx, tag=:tns) where {D, Tv, Ti}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSimpleJumpVector{Tv, Ti}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSimpleJumpVector) = tns.ex

function Finch.initialize!(arr::VirtualSimpleJumpVector{D, Tv}, ctx::Finch.LowerJulia, mode, idxs...) where {D, Tv}
    if mode.kind === updater
        push!(ctx.preamble, quote
            $(arr.ex).idx = [$(arr.ex).idx[end]]
            $(arr.ex).val = $Tv[]
        end)
    end
    access(arr, mode, idxs...)
end 

function Finch.getsize(arr::VirtualSimpleJumpVector{Tv, Ti}, ctx::Finch.LowerJulia, mode) where {Tv, Ti}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(literal(1), value(ex, Ti)),)
end
Finch.setsize!(arr::VirtualSimpleJumpVector{Tv, Ti}, ctx::Finch.LowerJulia, mode, dims...) where {Tv, Ti} = arr
Finch.getname(arr::VirtualSimpleJumpVector) = arr.name
Finch.setname(arr::VirtualSimpleJumpVector, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function Finch.stylize_access(node, ctx::Finch.Stylize{LowerJulia}, tns::VirtualSimpleJumpVector)
    if ctx.root isa IndexNode && ctx.root.kind === loop && ctx.root.idx == get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        Finch.DefaultStyle()
    end
end

function Finch.chunkify_access(node, ctx, vec::VirtualSimpleJumpVector{Tv, Ti}) where {Tv, Ti}
    my_I = ctx.ctx.freshen(getname(vec), :_I)
    my_i = ctx.ctx.freshen(getname(vec), :_i0)
    my_i′ = ctx.ctx.freshen(getname(vec), :_i1)
    my_p = ctx.ctx.freshen(getname(vec), :_p)
    if getname(ctx.idx) == getname(node.idxs[1])
        if node.mode.kind === reader
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
                        stride = (ctx, ext) -> value(my_i′),
                        body = (ctx, ext, ext_2) -> begin
                            Switch([
                                value(:($(ctx(getstop(ext_2))) == $my_i′)) => Thunk(
                                    body = Spike(
                                        body = Simplify(literal(zero(Tv))),
                                        tail = value(:($(vec.ex).val[$my_p]), Tv),
                                    ),
                                    epilogue = quote
                                        $my_p += 1
                                        $my_i = $my_i′ + 1
                                        $my_i′ = $(vec.ex).idx[$my_p]
                                    end
                                ),
                                literal(true) => Stepper(
                                    seek = (ctx, ext) -> quote
                                        $my_p = searchsortedfirst($(vec.ex).idx, $(ctx(getstart(ext))), $my_p, length($(vec.ex).idx), Base.Forward)
                                        $my_i = $(ctx(getstart(ext)))
                                        $my_i′ = $(vec.ex).idx[$my_p]
                                    end,
                                    body = Step(
                                        stride = (ctx, idx, ext) -> value(my_i′),
                                        chunk = Spike(
                                            body = Simplify(literal(zero(Tv))),
                                            tail = value(:($(vec.ex).val[$my_p]), Tv),
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
            return access(tns, node.mode, node.idxs...)
        else
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
                    body = value(:($(vec.ex).val[$my_p]), Tv),
                    epilogue = quote
                        $(vec.ex).idx[$my_p] = $(ctx(idx))
                    end
                )
            )
            return access(tns, node.mode, node.idxs...)
        end
    else
        node
    end
end

Finch.register()
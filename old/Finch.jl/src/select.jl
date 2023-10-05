struct SelectVisitor
    ctx
    idxs
end

function (ctx::SelectVisitor)(node)
    if istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        node
    end
end

function (ctx::SelectVisitor)(node::IndexNode)
    if node.kind === access && node.tns isa IndexNode && node.tns.kind === virtual
        select_access(node, ctx, node.tns.val)
    elseif istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        node
    end
end
select_access(node, ctx, tns) = similarterm(node, operation(node), map(ctx, arguments(node)))

struct SelectStyle end

combine_style(a::SelectStyle, b::ThunkStyle) = b
combine_style(a::SelectStyle, b::ChunkStyle) = a
combine_style(a::SelectStyle, b::SelectStyle) = a

function (ctx::LowerJulia)(root, ::SelectStyle)
    idxs = Dict()
    root = SelectVisitor(ctx, idxs)(root)
    for (idx, val) in pairs(idxs)
        ctx.dims[getname(idx)] = Extent(val, val)
        root = loop(idx, root)
    end
    contain(ctx) do ctx_2
        ctx_2(root)
    end
end

@kwdef struct Furlable
    name = gensym()
    size
    body
end

getsize(tns::Furlable, ::LowerJulia, mode) = tns.size
getname(tns::Furlable) = tns.name

IndexNotation.isliteral(::Furlable) = false

Base.show(io::IO, ex::Furlable) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Furlable)
    print(io, "Furlable()")
end

function stylize_access(node, ctx::Stylize{LowerJulia}, tns::Furlable)
    if !isempty(node.idxs)
        if getunbound(node.idxs[1]) ⊆ keys(ctx.ctx.bindings)
            return SelectStyle()
        elseif ctx.root isa IndexNode && ctx.root.kind === loop && ctx.root.idx == get_furl_root(node.idxs[1])
            return ChunkStyle()
        end
    end
    return DefaultStyle()
end

function select_access(node, ctx::Finch.SelectVisitor, tns::Furlable)
    if !isempty(node.idxs)
        if getunbound(node.idxs[1]) ⊆ keys(ctx.ctx.bindings)
            var = index(ctx.ctx.freshen(:s))
            val = cache!(ctx.ctx, :s, node.idxs[1])
            ctx.idxs[var] = val
            ext = first(getsize(tns, ctx.ctx, node.mode))
            ext_2 = Extent(val, val)
            tns_2 = truncate(tns, ctx.ctx, ext, ext_2)
            return access(tns_2, node.mode, var, node.idxs[2:end]...)
        end
    end
    return similarterm(node, operation(node), map(ctx, arguments(node)))
end

function chunkify_access(node, ctx, tns::Furlable)
    if !isempty(node.idxs)
        if ctx.idx == get_furl_root(node.idxs[1])
            idxs = map(ctx, node.idxs)
            return access(tns.body(ctx.ctx, ctx.idx, ctx.ext), node.mode, get_furl_root(node.idxs[1]), idxs[2:end]...)
        else
            idxs = map(ctx, node.idxs)
            return access(node.tns, node.mode, idxs...)
        end
    end
    return node
end

struct DiagMask end

const diagmask = DiagMask()

Base.show(io::IO, ex::DiagMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::DiagMask)
    print(io, "diagmask")
end

virtualize(ex, ::Type{DiagMask}, ctx) = diagmask
IndexNotation.isliteral(::DiagMask) = false
Finch.getname(::DiagMask) = gensym()
Finch.setname(::DiagMask, name) = diagmask
Finch.getsize(::DiagMask, ctx, mode) = (nodim, nodim)

function initialize!(::DiagMask, ctx, mode, idxs...)
    tns = Furlable(
        size = (nodim, nodim),
        body = (ctx, idx, ext) -> Lookup(
            body = (i) -> Furlable(
                size = (nodim,),
                body = (ctx, idx, ext) -> Pipeline([
                    Phase(
                        stride = (ctx, idx, ext) -> value(:($(ctx(i)) - 1)),
                        body = (start, step) -> Run(body=Simplify(Fill(false)))
                    ),
                    Phase(
                        stride = (ctx, idx, ext) -> i,
                        body = (start, step) -> Run(body=Simplify(Fill(true))),
                    ),
                    Phase(body = (start, step) -> Run(body=Simplify(Fill(false))))
                ])
            )
        )
    )
    return access(tns, mode, idxs...)
end

struct LoTriMask end

const lotrimask = LoTriMask()

Base.show(io::IO, ex::LoTriMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::LoTriMask)
    print(io, "lotrimask")
end

virtualize(ex, ::Type{LoTriMask}, ctx) = lotrimask
IndexNotation.isliteral(::LoTriMask) = false
Finch.getname(::LoTriMask) = gensym()
Finch.setname(::LoTriMask, name) = lotrimask
Finch.getsize(::LoTriMask, ctx, mode) = (nodim, nodim)

function initialize!(::LoTriMask, ctx, mode, idxs...)
    tns = Furlable(
        size = (nodim, nodim),
        body = (ctx, idx, ext) -> Lookup(
            body = (i) -> Furlable(
                size = (nodim,),
                body = (ctx, idx, ext) -> Pipeline([
                    Phase(
                        stride = (ctx, idx, ext) -> value(:($(ctx(i)))),
                        body = (start, step) -> Run(body=Simplify(Fill(true)))
                    ),
                    Phase(
                        body = (start, step) -> Run(body=Simplify(Fill(false))),
                    )
                ])
            )
        )
    )
    return access(tns, mode, idxs...)
end

struct UpTriMask end

const uptrimask = UpTriMask()

Base.show(io::IO, ex::UpTriMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::UpTriMask)
    print(io, "uptrimask")
end

virtualize(ex, ::Type{UpTriMask}, ctx) = uptrimask
IndexNotation.isliteral(::UpTriMask) = false
Finch.getname(::UpTriMask) = gensym()
Finch.setname(::UpTriMask, name) = uptrimask
Finch.getsize(::UpTriMask, ctx, mode) = (nodim, nodim)

function initialize!(::UpTriMask, ctx, mode, idxs...)
    tns = Furlable(
        size = (nodim, nodim),
        body = (ctx, idx, ext) -> Lookup(
            body = (i) -> Furlable(
                size = (nodim,),
                body = (ctx, idx, ext) -> Pipeline([
                    Phase(
                        stride = (ctx, idx, ext) -> value(:($(ctx(i)) - 1)),
                        body = (start, step) -> Run(body=Simplify(Fill(false)))
                    ),
                    Phase(
                        body = (start, step) -> Run(body=Simplify(Fill(true))),
                    )
                ])
            )
        )
    )
    return access(tns, mode, idxs...)
end

struct BandMask end

const bandmask = BandMask()

Base.show(io::IO, ex::BandMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::BandMask)
    print(io, "bandmask")
end

virtualize(ex, ::Type{BandMask}, ctx) = bandmask
IndexNotation.isliteral(::BandMask) = false
Finch.getname(::BandMask) = gensym()
Finch.setname(::BandMask, name) = bandmask
Finch.getsize(::BandMask, ctx, mode) = (nodim, nodim, nodim)

function initialize!(::BandMask, ctx, mode, idxs...)
    tns = Furlable(
        size = (nodim, nodim, nodim),
        body = (ctx, idx, ext) -> Lookup(
            body = (i) -> Furlable(
                size = (nodim, nodim),
                body = (ctx, idx, ext) -> Lookup(
                    body = (j) -> Furlable(
                        size = (nodim,),
                        body = (ctx, idx, ext) -> Pipeline([
                            Phase(
                                stride = (ctx, idx, ext) -> value(:($(ctx(i)) - 1)),
                                body = (start, step) -> Run(body=Simplify(Fill(false)))
                            ),
                            Phase(
                                stride = (ctx, idx, ext) -> j,
                                body = (start, step) -> Run(body=Simplify(Fill(true)))
                            ),
                            Phase(
                                body = (start, step) -> Run(body=Simplify(Fill(false))),
                            )
                        ])
                    )
                )
            )
        )
    )
    return access(tns, mode, idxs...)
end
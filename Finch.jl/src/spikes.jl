@kwdef struct Spike
    body
    tail
end

Base.show(io::IO, ex::Spike) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Spike)
    print(io, "Spike(body = ")
    print(io, ex.body)
    print(io, ")")
end

isliteral(::Spike) = false

struct SpikeStyle end

(ctx::Stylize{LowerJulia})(node::Spike) = SpikeStyle()
combine_style(a::DefaultStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::RunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::ThunkStyle, b::SpikeStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::SpikeStyle) = SimplifyStyle()
combine_style(a::AcceptRunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::SpikeStyle, b::SpikeStyle) = SpikeStyle()

function (ctx::LowerJulia)(root::Chunk, ::SpikeStyle)
    root_body = SpikeBodyVisitor(ctx, root.idx, root.ext, Extent(spike_body_getstop(getstop(root.ext), ctx), getstop(root.ext)))(root.body)
    if extent(root.ext) == 1
        body_expr = quote end
    else
        #TODO check body nonempty
        body_expr = contain(ctx) do ctx_2
            (ctx_2)(Chunk(
                idx = root.idx,
                ext = spike_body_range(root.ext, ctx),
                body = root_body,
            ))
        end
    end
    root_tail = SpikeTailVisitor(ctx, root.idx, getstop(root.ext))(root.body)
    tail_expr = contain(ctx) do ctx_2
        (ctx_2)(Chunk(
            idx = root.idx,
            ext = Extent(start = getstop(root.ext), stop = getstop(root.ext), lower = 1, upper = 1),
            body = root_tail,
        ))
    end
    return Expr(:block, body_expr, tail_expr)
end

@kwdef struct SpikeBodyVisitor
    ctx
    idx
    ext
    ext_2
end

function (ctx::SpikeBodyVisitor)(node)
    if istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        truncate(node, ctx.ctx, ctx.ext, ctx.ext_2)
    end
end

function (ctx::SpikeBodyVisitor)(node::Spike)
    return Run(node.body)
end

(ctx::SpikeBodyVisitor)(node::Shift) = Shift(SpikeBodyVisitor(;kwfields(ctx)..., ext = shiftdim(ctx.ext, call(-, node.delta)), ext_2 = shiftdim(ctx.ext_2, call(-, node.delta)))(node.body), node.delta)

spike_body_getstop(stop, ctx) = :($(ctx(stop)) - 1)
spike_body_getstop(stop::Integer, ctx) = stop - 1

spike_body_range(ext, ctx) = Extent(getstart(ext), spike_body_getstop(getstop(ext), ctx))

@kwdef struct SpikeTailVisitor
    ctx
    idx
    val
end

function (ctx::SpikeTailVisitor)(node)
    if istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        node
    end
end

(ctx::SpikeTailVisitor)(node::Access) = something(unchunk(node.tns, ctx), node)
unchunk(node::Spike, ctx::SpikeTailVisitor) = node.tail
unchunk(node::Shift, ctx::SpikeTailVisitor) = unchunk(node.body, SpikeTailVisitor(;kwfields(ctx)..., val = call(-, ctx.val, node.delta)))

#TODO this is sus
unchunk(node::Spike, ctx::ForLoopVisitor) = node.tail

supports_shift(::SpikeStyle) = true

@kwdef mutable struct AcceptSpike
    val
    tail
end

Base.show(io::IO, ex::AcceptSpike) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::AcceptSpike)
    print(io, "AcceptSpike(val = ")
    print(io, ex.val)
    print(io, ")")
end

default(node::AcceptSpike) = node.val

unchunk(node::AcceptSpike, ctx::ForLoopVisitor) = node.tail(ctx.ctx, ctx.val)

function truncate(node::Spike, ctx, ext, ext_2)
    return Cases([
        :($(ctx(getstop(ext_2))) < $(ctx(getstop(ext)))) => Run(node.body),
        true => node,
    ])
end
truncate_weak(node::Spike, ctx, ext, ext_2) = node
truncate_strong(node::Spike, ctx, ext, ext_2) = Run(node.body)
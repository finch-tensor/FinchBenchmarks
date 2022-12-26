struct ChunkStyle end

combine_style(a::DefaultStyle, b::ChunkStyle) = ChunkStyle()
combine_style(a::ThunkStyle, b::ChunkStyle) = ThunkStyle()
combine_style(a::ChunkStyle, b::ChunkStyle) = ChunkStyle()
combine_style(a::ChunkStyle, b::DimensionalizeStyle) = DimensionalizeStyle()
combine_style(a::ChunkStyle, b::SimplifyStyle) = SimplifyStyle()

struct ChunkifyVisitor
    ctx
    idx
    ext
end

function (ctx::ChunkifyVisitor)(node)
    if istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        node
    end
end

function (ctx::ChunkifyVisitor)(node::IndexNode)
    if node.kind === access && node.tns isa IndexNode && node.tns.kind === virtual
        chunkify_access(node, ctx, node.tns.val)
    elseif istree(node)
        similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        node
    end
end

chunkify_access(node, ctx, tns) = similarterm(node, operation(node), map(ctx, arguments(node)))

function (ctx::LowerJulia)(root::IndexNode, ::ChunkStyle)
    if root.kind === loop
        idx = root.idx
        #TODO is every read of dims gonna be like this? When do we lock it in?
        ext = resolvedim(ctx.dims[getname(idx)])
        body = (ChunkifyVisitor(ctx, idx, ext))(root.body)
        #TODO add a simplify step here perhaps
        ctx(chunk(
            idx,
            ext,
            body
        ))
    else
        error("unimplemented")
    end
end

#TODO one day this might be nothing?
truncate(node, ctx, ext, ext_2) = node
truncate_weak(node, ctx, ext, ext_2) = truncate(node, ctx, ext, ext_2)
truncate_strong(node, ctx, ext, ext_2) = truncate(node, ctx, ext, ext_2)
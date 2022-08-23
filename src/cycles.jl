function lower_cycle(root, ctx, idx, ext, style)
    i = getname(root.idx)
    i0 = ctx.freshen(i, :_start)
    push!(ctx.preamble, quote
        $i = $(ctx(getstart(root.ext)))
    end)

    guard = :($i <= $(ctx(getstop(root.ext))))
    body = CycleVisitor(style, ctx, idx, ext)(root.body)

    body_2 = contain(ctx) do ctx_2
        push!(ctx_2.preamble, :($i0 = $i))
        ctx_2(Chunk(root.idx, Extent(start = i0, stop = getstop(root.ext), lower = 1), body))
    end

    if simplify((@f $(getlower(ext)) >= 1)) == true  && simplify((@f $(getupper(ext)) <= 1)) == true
        body_2
    else
        return quote
            while $guard
                $body_2
            end
        end
    end
end

@kwdef struct CycleVisitor{Style}
    style::Style
    ctx
    idx
    ext
end

function (ctx::CycleVisitor)(node)
    if istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

(ctx::CycleVisitor)(node::Shift) = Shift(CycleVisitor(; kwfields(ctx)..., ext=shiftdim(ctx.ext, call(-, node.delta)))(node.body), node.delta)
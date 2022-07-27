struct Freshen
    seen
    counts
end
Freshen() = Freshen(Set(), Dict())
function (spc::Freshen)(tags...)
    name = Symbol(tags...)
    m = match(r"^(.*)_(\d*)$", string(name))
    if m === nothing
        tag = name
        n = 1
    else
        tag = Symbol(m.captures[1])
        n = parse(BigInt, m.captures[2])
    end
    if (tag, n) in spc.seen
        n = max(get(spc.counts, tag, 0), n) + 1
        spc.counts[tag] = n
    end
    push!(spc.seen, (tag, n))
    if n == 1
        return Symbol(tag)
    else
        return Symbol(tag, :_, n)
    end
end

@kwdef mutable struct LowerJulia <: AbstractVisitor
    preamble::Vector{Any} = []
    bindings::Dict{Any, Any} = Dict()
    epilogue::Vector{Any} = []
    dims::Dict = Dict()
    freshen::Freshen = Freshen()
end

function cache!(ctx, var, val)
    if isliteral(val)
        return val
    end
    body = contain(ctx) do ctx_2
        ctx(val)
    end
    if body isa Symbol
        return body
    else
        var = ctx.freshen(var)
        push!(ctx.preamble, Expr(:cache, var,
        quote
            $var = $body
        end))
        return Virtual{Any}(var)
    end
end

bind(f, ctx::LowerJulia) = f()
function bind(f, ctx::LowerJulia, (var, val′), tail...)
    if haskey(ctx.bindings, var)
        val = ctx.bindings[var]
        ctx.bindings[var] = val′
        res = bind(f, ctx, tail...)
        ctx.bindings[var] = val
        return res
    else
        ctx.bindings[var] = val′
        res = bind(f, ctx, tail...)
        pop!(ctx.bindings, var)
        return res
    end
end

function contain(f, ctx::LowerJulia)
    ctx_2 = shallowcopy(ctx)
    ctx_2.preamble = []
    ctx_2.epilogue = []
    body = f(ctx_2)
    thunk = Expr(:block)
    append!(thunk.args, ctx_2.preamble)
    if isempty(ctx_2.epilogue)
        push!(thunk.args, body)
    else
        res = ctx_2.freshen(:res)
        push!(thunk.args, Expr(:cleanup, res, body, Expr(:block, ctx_2.epilogue...)))
    end
    return thunk
end

struct ThunkStyle end

@kwdef struct Thunk
    preamble = quote end
    body
    epilogue = quote end
    binds = ()
end
isliteral(::Thunk) = false

Base.show(io::IO, ex::Thunk) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Thunk)
    print(io, "Thunk()")
end

(ctx::Stylize{LowerJulia})(node::Thunk) = ThunkStyle()
combine_style(a::DefaultStyle, b::ThunkStyle) = ThunkStyle()
combine_style(a::ThunkStyle, b::ThunkStyle) = ThunkStyle()

struct ThunkVisitor <: AbstractTransformVisitor
    ctx
end

function (ctx::LowerJulia)(node, ::ThunkStyle)
    contain(ctx) do ctx2
        node = (ThunkVisitor(ctx2))(node)
        (ctx2)(node)
    end
end

function (ctx::ThunkVisitor)(node::Thunk, ::DefaultStyle)
    push!(ctx.ctx.preamble, node.preamble)
    push!(ctx.ctx.epilogue, node.epilogue)
    for (var, val) in node.binds
        define!(ctx.ctx, var, val)
    end
    node.body
end

#default lowering

(ctx::LowerJulia)(::Pass, ::DefaultStyle) = quote end

function (ctx::LowerJulia)(root::Assign, ::DefaultStyle)
    if root.op == nothing
        rhs = ctx(root.rhs)
    else
        rhs = ctx(call(root.op, root.lhs, root.rhs))
    end
    lhs = ctx(root.lhs)
    :($lhs = $rhs)
end

function (ctx::LowerJulia)(root::Call, ::DefaultStyle)
    if root.op == and
        reduce((x, y) -> :($x && $y), map(ctx, root.args)) #TODO This could be better. should be able to handle empty case
    elseif root.op == or
        reduce((x, y) -> :($x || $y), map(ctx, root.args))
    else
        :($(ctx(root.op))($(map(ctx, root.args)...)))
    end
end

function (ctx::LowerJulia)(root::Name, ::DefaultStyle)
    @assert haskey(ctx.bindings, getname(root)) "variable $(getname(root)) unbound"
    return ctx(ctx.bindings[getname(root)]) #This unwraps indices that are virtuals. Arguably these virtuals should be precomputed, but whatevs.
end

function (ctx::LowerJulia)(root::Protocol, ::DefaultStyle)
    :($(ctx(root.idx)))
end

isliteral(::Union{Symbol, Expr, Missing}) = false
(ctx::LowerJulia)(root::Union{Symbol, Expr}, ::DefaultStyle) = root
(ctx::LowerJulia)(root::Literal{<:Union{Symbol, Expr, Missing}}, ::DefaultStyle) = QuoteNode(root.val)
(ctx::LowerJulia)(root::Literal, ::DefaultStyle) = root.val

function (ctx::LowerJulia)(root, ::DefaultStyle)
    if isliteral(root)
        return getvalue(root)
    end
    error("Don't know how to lower $root")
end

function (ctx::LowerJulia)(root::Virtual, ::DefaultStyle)
    return root.ex
end

function (ctx::LowerJulia)(root::With, ::DefaultStyle)
    prod = nothing
    target = map(getname, getresults(root.prod))
    return quote
        $(contain(ctx) do ctx_2
            prod = Initialize(ctx = ctx_2, target=target)(root.prod)
            (ctx_2)(prod)
        end)
        $(contain(ctx) do ctx_2
            Finalize(ctx = ctx_2, target=target)(prod)
            cons = Initialize(ctx = ctx_2, target=target)(root.cons)
            res = (ctx_2)(cons)
            Finalize(ctx = ctx_2, target=target)(cons)
            res
        end)
    end
end

function (ctx::LowerJulia)(root::Multi, ::DefaultStyle)
    thunk = Expr(:block)
    for body in root.bodies
        push!(thunk.args, quote
            $(contain(ctx) do ctx_2
                (ctx_2)(body)
            end)
        end)
    end
    thunk
end

function (ctx::LowerJulia)(root::Access, ::DefaultStyle)
    tns = ctx(root.tns)
    idxs = map(ctx, root.idxs)
    :($(ctx(tns))[$(idxs...)])
end


function (ctx::LowerJulia)(root::Access{<:Number, Read}, ::DefaultStyle)
    @assert isempty(root.idxs)
    return root.tns
end

function (ctx::LowerJulia)(stmt::Sieve, ::DefaultStyle)
    cond = ctx.freshen(:cond)
    push!(ctx.preamble, :($cond = $(ctx(stmt.cond))))
    body = ctx(stmt.body)

    return quote
        if $cond
            $body
        end
    end
end

function (ctx::LowerJulia)(stmt::Loop, ::DefaultStyle)
    ctx(Chunk(
        idx = stmt.idx,
        ext = resolvedim(ctx.dims[getname(stmt.idx)]),
        body = stmt.body)
    )
end
function (ctx::LowerJulia)(stmt::Chunk, ::DefaultStyle)
    idx_sym = ctx.freshen(getname(stmt.idx))
    if simplify((@i $(getlower(stmt.ext)) >= 1)) == true  && simplify((@i $(getupper(stmt.ext)) <= 1)) == true
        return quote
            $idx_sym = $(ctx(getstart(stmt.ext)))
            $(bind(ctx, getname(stmt.idx) => idx_sym) do 
                contain(ctx) do ctx_2
                    body_3 = ForLoopVisitor(ctx_2, stmt.idx, idx_sym)(stmt.body)
                    (ctx_2)(body_3)
                end
            end)
        end
    else
        return quote
            for $idx_sym = $(ctx(getstart(stmt.ext))):$(ctx(getstop(stmt.ext)))
                $(bind(ctx, getname(stmt.idx) => idx_sym) do 
                    contain(ctx) do ctx_2
                        body_3 = ForLoopVisitor(ctx_2, stmt.idx, idx_sym)(stmt.body)
                        (ctx_2)(body_3)
                    end
                end)
            end
        end
    end
end

@kwdef struct ForLoopVisitor <: AbstractTransformVisitor
    ctx
    idx
    val
end

@kwdef struct Leaf
    val = nothing
    body
end

default(ex::Leaf) = something(ex.val)

Base.show(io::IO, ex::Leaf) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Leaf)
    print(io, "Leaf()")
end

isliteral(node::Leaf) = false

function (ctx::ForLoopVisitor)(node::Access{Leaf}, ::DefaultStyle)
    node.tns.body(ctx.val)
end

function (ctx::ForLoopVisitor)(node::Leaf, ::DefaultStyle)
    node.body(ctx.val)
end

unchunk(node, ctx) = nothing
(ctx::ForLoopVisitor)(node::Access, ::DefaultStyle) = something(unchunk(node.tns, ctx), node)
unchunk(node::Leaf, ctx::ForLoopVisitor) = node.body(ctx.val)
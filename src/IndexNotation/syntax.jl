#TODO use MacroTools?

const incs = Dict(:+= => :+, :*= => :*, :&= => :&, :|= => :|)

const program_nodes = (
    pass = pass,
    loop = loop,
    chunk = chunk,
    with = with,
    sieve = sieve,
    multi = multi,
    assign = assign,
    call = call,
    access = access,
    protocol = protocol,
    name = Name,
    label = esc,
    value = esc,
)

const instance_nodes = (
    pass = pass_instance,
    loop = loop_instance,
    chunk = :(throw(NotImplementedError("TODO"))),
    with = with_instance,
    sieve = sieve_instance,
    multi = multi_instance,
    assign = assign_instance,
    call = call_instance,
    access = access_instance,
    protocol = protocol_instance,
    name = name_instance,
    label = (ex) -> :($label_instance($(QuoteNode(ex)), $value_instance($(esc(ex))))),
    value = (ex) -> :($value_instance($(esc(ex))))
)

and() = true
and(x) = x
and(x, y, tail...) = x && and(y, tail...)
or() = false
or(x) = x
or(x, y, tail...) = x || or(y, tail...)

function _finch_capture(ex, ctx)
    #extra sugar
    if ex isa Expr && ex.head == :macrocall && length(ex.args) >= 3 && ex.args[1] == Symbol("@∀")
        idxs = ex.args[3:end-1]; body = ex.args[end]
        return _finch_capture(:(@loop($(idxs...), $body)), ctx)
    elseif ex isa Expr && ex.head == :block
        bodies = filter(arg->!(arg isa LineNumberNode), ex.args)
        if length(bodies) == 1
            return _finch_capture(:($(bodies[1])), ctx)
        else
            return _finch_capture(:(@multi($(bodies...),)), ctx)
        end
    elseif ex isa Expr && haskey(incs, ex.head)
        (lhs, rhs) = ex.args; op = incs[ex.head]
        return _finch_capture(:($lhs << $op >>= $rhs), ctx)
    elseif ex isa Expr && ex.head == :comparison
        @assert length(ex.args) >= 3
        (a, cmp, b, tail...) = ex.args
        ex = :($cmp($a, $b))
        if isempty(tail)
            return _finch_capture(:($cmp($a, $b)), ctx)
        else
            return _finch_capture(:($cmp($a, $b) && $(Expr(:comparison, b, tail...))), ctx)
        end
    elseif ex isa Expr && ex.head == :&&
        (a, b) = ex.args
        return _finch_capture(:($and($a, $b)), ctx)
    elseif ex isa Expr && ex.head == :||
        (a, b) = ex.args
        return _finch_capture(:($or($a, $b)), ctx)
    end

    if @capture ex (@pass(args__))
        args = map(arg -> _finch_capture(arg, (ctx..., namify=false)), args)
        return :($(ctx.nodes.pass)($(args...)))
    elseif @capture ex (@sieve cond_ body_)
        cond = _finch_capture(cond, (ctx..., namify=true))
        body = _finch_capture(body, ctx)
        return :($(ctx.nodes.sieve)($cond, $body))
    elseif @capture ex (@loop idxs__ body_)
        idxs = map(idx -> _finch_capture(idx, (ctx..., namify=true)), idxs)
        body = _finch_capture(body, ctx)
        return :($(ctx.nodes.loop)($(idxs...), $body))
    elseif @capture ex (@chunk idx_ ext_ body_)
        idx = _finch_capture(idx, ctx)
        ext = _finch_capture(ext, (ctx..., namify=false))
        body = _finch_capture(body, ctx)
        return :($(ctx.nodes.chunk)($idx, $ext, $body))
    elseif @capture ex (cons_ where prod_)
        cons = _finch_capture(cons, ctx)
        prod = _finch_capture(prod, (ctx..., results=Set()))
        return :($(ctx.nodes.with)($cons, $prod))
    elseif @capture ex (@multi bodies__)
        bodies = map(arg -> _finch_capture(arg, ctx), bodies)
        return :($(ctx.nodes.multi)($(bodies...)))
    elseif @capture ex (lhs_ = rhs_)
        lhs = _finch_capture(lhs, (ctx..., mode=Write()))
        rhs = _finch_capture(rhs, ctx)
        return :($(ctx.nodes.assign)($lhs, $rhs))
    elseif @capture ex (lhs_ << op_ >>= rhs_)
        lhs = _finch_capture(lhs, (ctx..., mode=Update()))
        rhs = _finch_capture(rhs, ctx)
        op = _finch_capture(op, (ctx..., namify=false))
        return :($(ctx.nodes.assign)($lhs, $op, $rhs))
    elseif @capture ex (op_(args__))
        op = _finch_capture(op, (ctx..., namify=false, mode=Read()))
        args = map(arg->_finch_capture(arg, (ctx..., mode=Read())), args)
        return :($(ctx.nodes.call)($op, $(args...)))
    elseif @capture ex (tns_[idxs__])
        if ctx.mode isa Union{Write, Update} && tns isa Symbol
            push!(ctx.results, tns)
        end
        tns = _finch_capture(tns, (ctx..., namify=false, mode=Read()))
        idxs = map(idx->_finch_capture(idx, (ctx..., namify=true, mode=Read())), idxs)
        return :($(ctx.nodes.access)($tns, $(ctx.mode), $(idxs...)))
    elseif @capture ex (idx_::proto_)
        idx = _finch_capture(idx, ctx)
        return :($(ctx.nodes.protocol)($idx, $(esc(proto))))
    elseif ex isa Expr && ex.head == :$ && length(ex.args) == 1
        return esc(ex.args[1])
    elseif ex isa Symbol && ctx.namify
        return ctx.nodes.name(ex)
    elseif ex isa Symbol
        return ctx.nodes.label(ex)
    else
        return ctx.nodes.value(ex)
    end
end

capture_finch_program(ex; results=Set()) = _finch_capture(ex, (nodes=program_nodes, namify=true, mode = Read(), results = results))
capture_finch_instance(ex; results=Set()) = _finch_capture(ex, (nodes=instance_nodes, namify=true, mode = Read(), results = results))

macro finch_program(ex)
    return capture_finch_program(ex)
end

macro f(ex)
    return capture_finch_program(ex)
end

macro finch_program_instance(ex)
    return capture_finch_instance(ex)
end
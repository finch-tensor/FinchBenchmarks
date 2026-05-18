using Finch
using BenchmarkTools
using SparseArrays


function coalesce_impl(v, num_cpu)
    dev = cpu(:t, num_cpu)
    _s = Tensor(Coalesce(dev, Element(0.0)))
    _v = Tensor(SparseList(Element(0.0)), v)

    time = @belapsed begin
        (_s, _v, dev) = $(_s, _v, dev)
        @finch mode = :fast begin
            _s .= 0
            for i = parallel(_, dev)
                _s[] += _v[i]
            end
        end
    end

    @finch mode = :fast begin
        _s .= 0
        for i = parallel(_, dev)
            _s[] += _v[i]
        end
    end

    return (; time=time, s=_s[])
end
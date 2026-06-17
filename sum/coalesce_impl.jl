using Finch
using BenchmarkTools
using SparseArrays


function coalesce_impl(v1, v2, num_cpu)
    dev = cpu(:t, num_cpu)
    _s = Tensor(Coalesce(dev, Element(0.0)))
    _v1 = Tensor(SparseList(Element(0.0)), v1)
    _v2 = Tensor(SparseList(Element(0.0)), v2)

    time = @belapsed begin
        (_s, _v1, _v2, dev) = $(_s, _v1, _v2, dev)
        @finch mode = :fast begin
            _s .= 0
            for i = parallel(_, dev)
                _s[] += _v1[i] * _v2
            end
        end
    end

    _s = Tensor(Coalesce(dev, Element(0.0)))
    @finch mode = :fast begin
        _s .= 0
        for i = parallel(_, dev)
            _s[] += _v1[i] * _v2[i]
        end
    end

    return (; time=time, s=_s[])
end
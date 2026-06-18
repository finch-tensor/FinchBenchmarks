using Finch
using BenchmarkTools
using SparseArrays


function finch_impl(B, C, D, num_cpu)
    dev = cpu(:t, num_cpu)
    _B = Tensor(SparseList(SparseList(SparseList(Element(0.0)))), B)
    _C = Tensor(Dense(Dense(Element(0.0))), C)
    _D = Tensor(Dense(Dense(Element(0.0))), D)
    
    dev = cpu(:t, num_cpu)
    _A = Tensor(Dense(Shard(dev, SparseList(Element(0.0)))))
    # _A = Tensor(Dense(Dense(Element(0.0))), size(_B)[1], size(_C)[2])
    time = @belapsed begin
        (_A, _B, _C, _D, dev) = $(_A, _B, _C, _D, dev)
        @finch mode = :fast begin
            _A .= 0
           for r = parallel(_, dev)
                for k = _
                    for j = _
                        for i = _
                            _A[i, r] += _B[i, j, k] * _D[k, r] * _C[j, r]
                        end
                    end
                end
            end
        end
    end

    # for some unknown reason, the belapsed block affects the variables outside of it
   # _A = Tensor(Dense(Dense(Element(0.0))))

    #@finch begin
     #   _A .= 0
       # for r = parallel(_)
#	 for r=_
 #           for k = _
  #              for j = _
   #                 for i = _
    #                    _A[i, r] += _B[i, j, k] * _D[k, r] * _C[j, r]
     #               end
      #          end
       #     end
       # end
   # end

    return (; time=time, A=_A)
end

using Finch
using BenchmarkTools


function parallel_col_separate_sparselist_results_add(A, B, num_cpu)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _B = Tensor(Dense(SparseList(Element(0.0))), B)

        time = @belapsed begin
                _C = Tensor(Dense(Separate(SparseList(Element(0.0)))))
                kernel!($(_A), $(_B), _C)
        end
        
        _C = Tensor(Dense(Separate(SparseList(Element(0.0)))))
        kernel!(_A, _B, _C)
        return (; time=time, C=_C)
end


function kernel!(_A, _B, _C)
        @finch mode = :fast begin
                _C .= 0
                for j = parallel(_), i = _
                        _C[i, j] = _A[i, j] + _B[i, j]
                end
        end
end
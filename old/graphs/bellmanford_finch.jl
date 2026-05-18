"""
    bellmanford(adj, source=1)

Calculate the shortest paths from the vertex `source` in the graph specified by
an adjacency matrix `adj`, whose entries are edge weights. Weights should be
infinite when unconnected.

The output is given as a vector of distance, parent pairs for each node in the graph.
"""
function bellmanford_finch_kernel(edges, source=1)
    (n, m) = size(edges)
    @assert n == m

    dists_prev = Tensor(Dense(Element(Inf)), n)
    dists_prev[source] = 0
    dists = Tensor(Dense(Element(Inf)), n)
    active_prev = Tensor(SparseByteMap(Pattern()), n)
    active_prev[source] = true
    active = Tensor(SparseByteMap(Pattern()), n)
    parents = Tensor(Dense(Element(0)), n)
    any_active = Scalar(false)

    for iter = 1:n  
        @finch for j=_; if active_prev[j] dists[j] <<min>>= dists_prev[j] end end

        @finch begin
            active .= false
            for j = _
                if active_prev[j]
                    for i = _
                        let d = dists_prev[j] + edges[i, j]
                            dists[i] <<min>>= d
                            active[i] |= d < dists_prev[i]
                        end
                    end
                end
            end
        end

        iter_max = iter


        @finch begin
            any_active .= false
            for i = _
                any_active[] |= active[i]
            end
        end
        if !any_active[]
            break
        end
        dists_prev, dists = dists, dists_prev
        active_prev, active = active, active_prev
    end

    @finch begin
        for j = _
            for i = _
                let d = edges[i, j]
                    if d < Inf && dists[j] + d <= dists[i]
                        parents[i] <<choose(0)>>= j
                    end
                end
            end
        end
    end

    return (dists=dists, parents=parents)
end

function bellmanford_finch(mtx)
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)
    time = @belapsed bellmanford_finch_kernel($A, 1)
    output = bellmanford_finch_kernel(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end


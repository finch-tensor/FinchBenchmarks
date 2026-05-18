using TensorMarket
function bfs_lagraph(A)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    fwrite(A_path, Tensor(CSCFormat(fill_value(A)), A)) #TACO matrix market readerr can only read real-valued matrices
    lagraph_path = joinpath(@__DIR__, "../deps/LAGraph/build/src/benchmark/bfs_demo")
    output = withenv("OMP_NUM_THREADS"=>"1") do
        read(pipeline(`$lagraph_path $A_path`), String)
    end
    time = match(r"Avg: BFS pushpull parent only\s+threads\s+1:\s+(\S+)\s+sec:", output).captures[1]
    return (;time=time, mem = Base.summarysize(A), output=nothing)
end

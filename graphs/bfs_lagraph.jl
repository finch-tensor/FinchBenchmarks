using Finch
using TensorMarket
using JSON
function bfs_lagraph(A)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    parents_path = joinpath(tmpdir, "parents.mtx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    lagraph_path = joinpath(@__DIR__, "../deps/LAGraph/build/src")
    lagraphx_path = joinpath(@__DIR__, "../deps/LAGraph/build/experimental")
    graphblas_path = joinpath(@__DIR__, "../deps/GraphBLAS/build")
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$lagraph_path:$lagraphx_path:$graphblas_path", "LD_LIBRARY_PATH" => "$lagraph_path:$lagraphx_path:$graphblas_path", "OMP_NUM_THREADS"=>"1") do
        bfs_path = joinpath(@__DIR__, "bfs_lagraph")
        run(`$bfs_path -i $tmpdir -o $tmpdir`)
    end
    parents = Array(fread(parents_path)[:,1]) .+ 1
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, mem = Base.summarysize(A), output=parents)
end
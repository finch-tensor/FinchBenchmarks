using Finch
using TensorMarket
using JSON
function bellmanford_lagraph(A)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    distances_path = joinpath(tmpdir, "distances.ttx")
    parents_path = joinpath(tmpdir, "parents.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(fill_value(A))), A))
    lagraph_path = joinpath(@__DIR__, "../deps/LAGraph/build/src")
    lagraphx_path = joinpath(@__DIR__, "../deps/LAGraph/build/experimental")
    graphblas_path = joinpath(@__DIR__, "../deps/GraphBLAS/build")
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$lagraph_path:$lagraphx_path:$graphblas_path", "LD_LIBRARY_PATH" => "$lagraph_path:$lagraphx_path:$graphblas_path", "OMP_NUM_THREADS"=>"1") do
        bellmanford_path = joinpath(@__DIR__, "bellmanford_lagraph")
        run(`$bellmanford_path -i $tmpdir -o $tmpdir -- $args`)
    end
    distances = Vector(reshape(fread(distances_path), :))
    parents = Vector(reshape(fread(parents_path), :))
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, output=(;dists=distances, parents=parents))
end
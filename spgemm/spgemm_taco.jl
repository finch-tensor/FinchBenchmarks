using Finch
using TensorMarket
using JSON
function spgemm_taco(args, A, B)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    # C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
    taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
    compiler = Sys.isapple() ? "gcc-14" : "gcc"
    num_threads = Threads.nthreads()
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CC" => "$compiler", "OMP_NUM_THREADS" => "$num_threads") do
        spgemm_path = joinpath(@__DIR__, "spgemm_taco")
        run(`$spgemm_path -i $tmpdir -o $tmpdir -- $args`)
    end
    # C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    # return (;time=time*10^-9, C=C)
    return (;time=time*10^-9,)
end

spgemm_taco_inner(A, B) = spgemm_taco(`--schedule inner`, A, permutedims(B))
spgemm_taco_gustavson(A, B) = spgemm_taco(`--schedule gustavson`, A, B)
spgemm_taco_outer(A, B) = spgemm_taco(`--schedule outer`, permutedims(A), B)

has_taco() = isfile(joinpath(@__DIR__, "spgemm_taco"))

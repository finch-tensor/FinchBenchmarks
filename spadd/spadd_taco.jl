using Finch
using TensorMarket
using JSON
function spadd_taco_helper(A, B)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
    taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        spadd_path = joinpath(@__DIR__, "spadd_taco")
        run(`$spadd_path -i $tmpdir -o $tmpdir`)
    end
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

spadd_taco(C, A, B) = spadd_taco_helper(A, B)

has_taco() = isfile(joinpath(@__DIR__, "spadd_taco"))

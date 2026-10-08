using Finch
using TensorMarket
using JSON
using SparseArrays

# variant 0-7 selects the transpose path in outer_mkl.cpp; all compute C = A * B^T.
function outer_mkl(A, B, nt; variant=0)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market reader can only read real-valued matrices
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
    mklvars_path = "/opt/intel/oneapi/setvars.sh"
    outer_path = joinpath(@__DIR__, "outer_mkl")
    withenv() do
        cmd = "source $mklvars_path; $outer_path -i $tmpdir -o $tmpdir $variant"
        run(`bash -c $cmd`)
    end
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

has_mkl() = isfile(joinpath(@__DIR__, "outer_mkl"))

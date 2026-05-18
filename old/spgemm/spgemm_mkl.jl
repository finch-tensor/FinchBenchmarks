using Finch
using TensorMarket
using JSON
function spgemm_mkl(A, B)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
    mklvars_path = joinpath(@__DIR__, "../deps/intel/setvars.sh")
    spgemm_path = joinpath(@__DIR__, "spgemm_mkl")
    withenv() do
        cmd = "source $mklvars_path; $spgemm_path -i $tmpdir -o $tmpdir"
        run(`bash -c $cmd`)
    end 
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

has_mkl() = isfile(joinpath(@__DIR__, "spgemm_mkl"))
using Finch
using TensorMarket
using JSON
function spmv_mkl(y, A, x)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    x_path = joinpath(tmpdir, "x.ttx")
    y_path = joinpath(tmpdir, "y.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    fwrite(x_path, Tensor(Dense(SparseList(Element(0.0))), reshape(Vector(x), :, 1)))
    mklvars_path = joinpath(@__DIR__, "../deps/intel/setvars.sh")
    spmv_path = joinpath(@__DIR__, "spmv_mkl")
    withenv() do
        cmd = "source $mklvars_path; $spmv_path -i $tmpdir -o $tmpdir"
        run(`bash -c $cmd`)
    end 
    y = fread(y_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, y=y)
end

has_mkl() = isfile(joinpath(@__DIR__, "spmv_mkl"))

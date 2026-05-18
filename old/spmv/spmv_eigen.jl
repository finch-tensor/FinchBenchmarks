using Finch
using TensorMarket
using JSON

function spmv_eigen(y, A, x)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    x_path = joinpath(tmpdir, "x.ttx")
    y_path = joinpath(tmpdir, "y.ttx")
    
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    fwrite(x_path, Tensor(Dense(SparseList(Element(0.0))), reshape(Vector(x), :, 1)))
    
    spmv_path = joinpath(@__DIR__, "spmv_eigen")
    withenv() do
        run(`$spmv_path -i $tmpdir -o $tmpdir`)
    end 
    
    y = Vector(reshape(SparseMatrixCSC(fread(y_path)), :))
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    
    return (;time=time*10^-9, y=y)
end

has_eigen() = isfile(joinpath(@__DIR__, "spmv_eigen"))
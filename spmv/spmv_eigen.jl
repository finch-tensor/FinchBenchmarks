using Finch
using TensorMarket
using JSON
function spmv_eigen(y, A, x)
    #mktempdir(prefix="input_") do tmpdir
    let tmpdir="."
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
	spmv_path = joinpath(@__DIR__, "spmv_eigen")
	withenv() do
	  run(`$spmv_path -i $tmpdir -o $tmpdir`)
	end 
        y = fread(y_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, y=y)
    end
end

using Finch
using TensorMarket
using JSON
function spgemm_eigen(A, B)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
    spgemm_path = joinpath(@__DIR__, "spgemm_eigen")
    withenv() do
        run(`$spgemm_path -i $tmpdir -o $tmpdir`)
    end 
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

has_eigen() = isfile(joinpath(@__DIR__, "spgemm_eigen"))
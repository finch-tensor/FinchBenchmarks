using Finch
using TensorMarket
using JSON

function hadamard_eigen_helper(A, B, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
        hadamard_path = joinpath(@__DIR__, "hadamard_eigen")
        run(`$hadamard_path -i $tmpdir -o $tmpdir -- -t $num_cpu`)
        C = fread(C_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, C = C)
    end
end

eigen_impl(A, B, num_cpu) = hadamard_eigen_helper(A, B, num_cpu)

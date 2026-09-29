using Finch
using TensorMarket
using JSON

# note: -t is taken but not used by nacho's kernels. nacho reads cpu count directly
function spadd_nacho_helper(A, B, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
        spadd_path = joinpath(@__DIR__, "spadd_nacho")
        run(`$spadd_path -i $tmpdir -o $tmpdir`)
        C = fread(C_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, C = C)
    end
end

nacho_dcsr_impl(A, B, num_cpu) = spadd_nacho_helper(A, B, num_cpu)

using Finch
using TensorMarket
using JSON

function spadd_mkl_helper(A, B, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, A)
        fwrite(B_path, B)
        spadd_path = joinpath(@__DIR__, "mkl_kernel")
        run(`$spadd_path -i $tmpdir -o $tmpdir -- -t $num_cpu`)
        C = fread(C_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, C = C)
    end
end

mkl_impl(A, B, num_cpu) = spadd_mkl_helper(A, B, num_cpu)
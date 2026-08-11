using Finch
using JSON
using TensorMarket


function add_halide_helper(num_cpu, A, B)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        fwrite(A_path, A)
        fwrite(B_path, B)
        C_path = joinpath(tmpdir, "C.ttx")

        halide_path = joinpath(@__DIR__, "halide_kernel")
        cmd = `$halide_path -i $tmpdir -o $tmpdir -- -t $num_cpu`
        run(cmd)

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        C = fread(C_path)
        return (;time=parsed["time"]*10^-9, C=C)
    end
end

add_halide_impl(A, B, num_cpu) = add_halide_helper(num_cpu, A, B)

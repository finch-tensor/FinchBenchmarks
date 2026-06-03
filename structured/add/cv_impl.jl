using Finch
using JSON
using TensorMarket


function add_cv_helper(args, num_cpu, A, B)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        fwrite(A_path, A)
        fwrite(B_path, B)
        C_path = joinpath(tmpdir, "C.ttx")

        opencv_path = joinpath(@__DIR__, "cv_kernel")
        cmd = `$opencv_path -i $tmpdir -o $tmpdir -t $num_cpu $args`
        run(cmd)

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        C = fread(C_path)
        return (;time=parsed["time"]*10^-9, C=C)
    end
end

add_cv_impl(A, B, num_cpu) = add_cv_helper("", num_cpu, A, B)

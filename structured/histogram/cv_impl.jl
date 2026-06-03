using Finch
using JSON
using TensorMarket


function hist_cv_helper(args, num_cpu, A)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        fwrite(A_path, A)
        hist_path = joinpath(tmpdir, "hist.ttx")

        opencv_path = joinpath(@__DIR__, "cv_kernel")
        cmd = `$opencv_path -i $tmpdir -o $tmpdir -t $num_cpu $args`
        run(cmd)

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        hist = fread(hist_path)
        return (;time=parsed["time"]*10^-9, hist=hist)
    end
end

hist_cv_impl(A, num_cpu) = hist_cv_helper("", num_cpu, A)

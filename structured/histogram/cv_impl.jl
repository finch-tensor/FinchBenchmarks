using Finch
using JSON
using TensorMarket


function hist_cv_helper(args, num_cpu, A)
    mktempdir(prefix="input_") do tmpdir
        R_path = joinpath(tmpdir, "R.ttx")
        G_path = joinpath(tmpdir, "G.ttx")
        B_path = joinpath(tmpdir, "B.ttx")

        m, n = size(A)
        R = fill(UInt8(0), m, n)
        G = fill(UInt8(0), m, n)
        B = fill(UInt8(0), m, n)

        for i in 1:m
            for j in 1:n
                R[i, j] = A[i, j][1]
                G[i, j] = A[i, j][2]
                B[i, j] = A[i, j][3]
            end
        end
        fwrite(R_path, R)
        fwrite(G_path, G)
        fwrite(B_path, B)
        hist_path = joinpath(tmpdir, "hist.ttx")

        opencv_path = joinpath(@__DIR__, "cv_kernel")
        cmd = `$opencv_path -i $tmpdir -o $tmpdir -- -t $num_cpu $args`
        run(cmd)

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        hist = fread(hist_path)
        return (;time=parsed["time"]*10^-9, hist=hist)
    end
end

hist_cv_impl(A, num_cpu) = hist_cv_helper("", num_cpu, A)

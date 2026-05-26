function sum_eigen_helper(B, C, D, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        D_path = joinpath(tmpdir, "D.ttx")
        fwrite(B_path, B)
        fwrite(C_path, C)
        fwrite(D_path, D)

        mttkrp_path = joinpath(@__DIR__, "eigen_kernel")
        run(`$mttkrp_path -i $tmpdir -o $tmpdir -t $num_cpu`)

        A_tensor = fread(A_path)

        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, A = A_tensor)
    end
end

eigen_impl(B, C, D, num_cpu) = sum_eigen_helper(B, C, D, num_cpu)
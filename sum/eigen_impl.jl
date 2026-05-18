function sum_eigen_helper(v, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        v_path = joinpath(tmpdir, "v.ttx")
        s_path = joinpath(tmpdir, "s.ttx")
        fwrite(v_path, Tensor(SparseList(Element(0.0)), v))

        sum_path = joinpath(@__DIR__, "eigen_kernel")
        run(`$sum_path -i $tmpdir -o $tmpdir -- -t $(num_cpu)`)

        # s = fread(s_path)
        s_tensor = fread(s_path)
        s = s_tensor[1, 1]

        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, s = s)
    end
end

eigen_impl(v, num_cpu) = sum_eigen_helper(v, num_cpu)
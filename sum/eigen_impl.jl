function sum_eigen_helper(v1, v2, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        v1_path = joinpath(tmpdir, "v1.ttx")
        v2_path = joinpath(tmpdir, "v2.ttx")
        s_path = joinpath(tmpdir, "s.ttx")
        fwrite(v1_path, Tensor(SparseList(Element(0.0)), v1))
        fwrite(v2_path, Tensor(SparseList(Element(0.0)), v2))

        sum_path = joinpath(@__DIR__, "sum_eigen")
        run(`$sum_path -i $tmpdir -o $tmpdir -- -t $(num_cpu)`)

        # s = fread(s_path)
        s_tensor = fread(s_path)
        s = s_tensor[1, 1]

        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, s = s)
    end
end

eigen_impl(v1, v2, num_cpu) = sum_eigen_helper(v1, v2, num_cpu)
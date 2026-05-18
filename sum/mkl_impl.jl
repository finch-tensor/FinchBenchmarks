function sum_mkl_helper(args, v, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        v_path = joinpath(tmpdir, "v.ttx")
        s_path = joinpath(tmpdir, "s.ttx")
        fwrite(v_path, Tensor(SparseList(Element(0.0)), v))

        # mkl_root = get(ENV, "MKL_ROOT", "/usr")
        # mkl_path = joinpath(mkl_root, "lib", "x86_64-linux-gnu")

        sum_path = joinpath(@__DIR__, "mkl_kernel")
        run(`$sum_path -i $tmpdir -o $tmpdir -- -t $(num_cpu)`)


        s = open(s_path) do f
            for line in eachline(f)
                startswith(line, '%') && continue  # skip %% and % lines
                stripped = strip(line)
                isempty(stripped) && continue      # skip the blank dimension line
                return parse(Float64, stripped)    # first non-comment, non-empty line is the value
            end
        end
        # s_tensor = fread(s_path)
        # s = s_tensor[]
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time = time * 10^-9, s = s)
    end
end

mkl_impl(v, num_cpu) = sum_mkl_helper("", v, num_cpu)
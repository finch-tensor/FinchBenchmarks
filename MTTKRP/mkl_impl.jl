function mttkrp_mkl_helper(args, B, C, D, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        D_path = joinpath(tmpdir, "D.ttx")
        fwrite(B_path, B)
        fwrite(C_path, C)
        fwrite(D_path, D)

        mttkrp_path = joinpath(@__DIR__, "mttkrp_mkl")
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        mklvars_path = joinpath(@__DIR__, "../deps/intel/oneapi/setvars.sh")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path","LD_LIBRARY_PATH" => "$taco_path") do
            cmd = "source $mklvars_path; $mttkrp_path -i $tmpdir -o $tmpdir"
            run(`bash -c $cmd`)
        end

        A_tensor = fread(A_path)
        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        return (;time=parsed["time"] * 10^-9, A=A_tensor)
    end
end

mkl_impl(B, C, D, num_cpu) = mttkrp_mkl_helper("", B, C, D, num_cpu)
using Finch
using TensorMarket
using JSON

function mttkrp_taco_helper(args, num_cpu, B, C, D)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        D_path = joinpath(tmpdir, "D.ttx")
        fwrite(B_path, B)
        fwrite(C_path, C)
        fwrite(D_path, D)
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", 
                "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb", 
                "OMP_NUM_THREADS" => string(num_cpu)) do

            mttkrp_path = joinpath(@__DIR__, "taco_kernel")
            cmd = isempty(args) ? `$mttkrp_path -i $tmpdir -o $tmpdir` : `$mttkrp_path -i $tmpdir -o $tmpdir $args`
            run(cmd)
        end
        A_tensor = fread(A_path)

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        return (;time=parsed["time"]*10^-9, A=A_tensor)
    end
end

taco_impl(B, C, D, num_cpu) = mttkrp_taco_helper("", num_cpu, B, C, D)

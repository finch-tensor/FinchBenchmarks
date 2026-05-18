using Finch
using TensorMarket
using JSON
function sum_taco_helper(args, num_cpu, v)
    mktempdir(prefix="input_") do tmpdir
        v_path = joinpath(tmpdir, "v.ttx")
        s_path = joinpath(tmpdir, "s.ttx")
        fwrite(v_path, v)
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", 
                "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb",
                "OMP_NUM_THREADS" => string(num_cpu)) do
                
            sum_path = joinpath(@__DIR__, "taco_kernel")
            cmd = isempty(args) ? `$sum_path -i $tmpdir -o $tmpdir` : `$sum_path -i $tmpdir -o $tmpdir $args`
            run(cmd)
        end
        # s_tensor = fread(s_path)
        # s = s_tensor[]

        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        # time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=parsed["time"]*10^-9, s=parsed["result"])
    end
end

taco_impl(v, num_cpu) = sum_taco_helper("", num_cpu, v)

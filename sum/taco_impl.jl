using Finch
using TensorMarket
using JSON
function sum_taco_helper(args, num_cpu, v1, v2)
    mktempdir(prefix="input_") do tmpdir
        v1_path = joinpath(tmpdir, "v1.ttx")
        v2_path = joinpath(tmpdir, "v2.ttx")
        s_path = joinpath(tmpdir, "s.ttx")
        fwrite(v1_path, v1)
        fwrite(v2_path, v2)
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", 
                "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb",
                "OMP_NUM_THREADS" => string(num_cpu)) do
                
            sum_path = joinpath(@__DIR__, "sum_taco")
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

taco_impl(v1, v2, num_cpu) = sum_taco_helper("", num_cpu, v1, v2)

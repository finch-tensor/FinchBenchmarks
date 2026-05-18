using Finch
using TensorMarket
using JSON

function spadd_taco_helper(args, A, B)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        fwrite(A_path, A)
        fwrite(B_path, B)
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
            spadd_path = joinpath(@__DIR__, "taco_kernel")
            cmd = isempty(args) ? `$spadd_path -i $tmpdir -o $tmpdir` : `$spadd_path -i $tmpdir -o $tmpdir $args`
            run(cmd)
        end
        
        parsed = JSON.parsefile(joinpath(tmpdir, "measurements.json"))
        return (;time=parsed["time"]*10^-9, C=fread(joinpath(tmpdir, "C.ttx")))
    end
end

taco_impl(A, B, num_cpu) = spadd_taco_helper("", A, B)
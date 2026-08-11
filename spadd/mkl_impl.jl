using Finch
using TensorMarket
using JSON

function spadd_mkl_helper(A, B, num_cpu)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B))
        spadd_path = joinpath(@__DIR__, "spadd_mkl")
        mklvars_path = "/opt/intel/oneapi/setvars.sh"
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path") do
            cmd = "source $mklvars_path; $spadd_path -i $tmpdir -o $tmpdir"
            run(`bash -c $cmd`)
        end
        C = fread(C_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (; time=time * 10^-9, C=C)
    end

    # tmpdir = mktempdir(prefix="input_", cleanup=false)
    # C_path = joinpath(tmpdir, "C.ttx")
    # @info "tmpdir: $tmpdir"
    # fwrite(joinpath(tmpdir, "A.ttx"), A)
    # fwrite(joinpath(tmpdir, "B.ttx"), B)
    # @info "files: $(readdir(tmpdir))"
    # spadd_path = joinpath(@__DIR__, "mkl_kernel")
    # run(`$spadd_path -i $tmpdir -o $tmpdir -t $num_cpu`)
    # C = fread(C_path)
    # time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    # return (; time = time * 10^-9, C = C)
end

mkl_impl(A, B, num_cpu) = spadd_mkl_helper(A, B, num_cpu)

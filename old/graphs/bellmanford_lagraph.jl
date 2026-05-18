function bellmanford_lagraph(A)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    fwrite(A_path, Tensor(CSCFormat(fill_value(A)), A)) #TACO matrix market readerr can only read real-valued matrices
    lagraph_path = joinpath(@__DIR__, "../deps/LAGraph/build/experimental/test/test_BF")
    output = withenv("MATRIX_INPUT"=>"./rmat_s4_e2.mtx") do
        err = IOBuffer();
        out = read(pipeline(ignorestatus(`../deps/LAGraph/build/experimental/test/test_BF`), stderr=err),String)
        String(take!(err)) * out
    end
    time = match(r"BF_full1a\s+time:\s+(\S+)\s+\(sec\)", output).captures[1]
    return (;time=time, mem = Base.summarysize(A), output=nothing)
end

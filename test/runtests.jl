using Finch
using Test
using SyntaxInterface
using Base.Iterators

include("data_matrices.jl")

function diff(name, res)
    global ARGS
    "nodiff" in ARGS && return true
    ref_dir = joinpath(@__DIR__, "reference")
    ref_file = joinpath(ref_dir, name)
    if "overwrite" in ARGS
        mkpath(ref_dir)
        open(ref_file, "w") do f
            println(f, res)
        end
        true
    else
        ref = read(ref_file, String)
        res = sprint(println, res)
        if ref == res
            return true
        else
            if "verbose" in ARGS
                println("=== reference ===")
                println(ref)
                println("=== test ===")
                println(res)
            end
            return false
        end
    end
end

reference_getindex(arr, inds...) = getindex(arr, inds...)
reference_getindex(arr::Fiber, inds...) = arr(inds...)

function reference_isequal(a,b)
    size(a) == size(b) || return false
    axes(a) == axes(b) || return false
    for i in Base.product(axes(a)...)
        reference_getindex(a, i...) == reference_getindex(b, i...) || return false
    end
    return true
end

using Finch: VirtualAbstractArray, Run, Spike, Extent, Scalar, Switch, Stepper, Jumper, Step, Jump, AcceptRun, AcceptSpike, Thunk, Phase, Pipeline, Lookup, Simplify, Shift
using Finch: @f, @finch_program_instance, execute, execute_code, getstart, getstop
using Finch: getname, value
using Finch.IndexNotation
using Finch.IndexNotation: call_instance, assign_instance, access_instance, value_instance, index_instance, loop_instance, with_instance, label_instance, protocol_instance

isstructequal(a, b) = a === b

isstructequal(a::T, b::T) where {T <: Fiber} = 
    isstructequal(a.lvl, b.lvl) &&
    isstructequal(a.env, b.env)

isstructequal(a::T, b::T)  where {T <: Pattern} = true

isstructequal(a::T, b::T) where {T <: Element} =
    a.val == b.val

isstructequal(a::T, b::T) where {T <: RepeatRLE} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.idx == b.idx &&
    a.val == b.val

isstructequal(a::T, b::T) where {T <: RepeatRLEDiff} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.idx == b.idx &&
    a.val == b.val

isstructequal(a::T, b::T) where {T <: Dense} =
    a.I == b.I &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseList} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.idx == b.idx &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseListDiff} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.idx == b.idx &&
    a.jdx == b.jdx &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseHash} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.tbl == b.tbl &&
    a.srt == b.srt &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseCoo} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.tbl == b.tbl &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseVBL} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.idx == b.idx &&
    a.ofs == b.ofs &&
    isstructequal(a.lvl, b.lvl)

isstructequal(a::T, b::T) where {T <: SparseBytemap} =
    a.I == b.I &&
    a.pos == b.pos &&
    a.tbl == b.tbl &&
    a.srt == b.srt &&
    a.srt_stop[] == b.srt_stop[] &&
    isstructequal(a.lvl, b.lvl)

verbose = "verbose" in ARGS

@testset "Finch.jl" begin
    include("test_util.jl")
    include("test_ssa.jl")
    include("test_print.jl")
    #include("test_parse.jl")
    include("test_formats.jl")
    include("test_constructors.jl")
    include("test_conversions.jl")
    include("test_merges.jl")
    include("test_algebra.jl")
    include("test_repeat.jl")
    include("test_permit.jl")
    include("test_skips.jl")
    include("test_fibers.jl")
    include("test_kernels.jl")
    include("test_issues.jl")
    include("test_packbits.jl")
end
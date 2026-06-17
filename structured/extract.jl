using Pkg

Pkg.add("FileIO")
Pkg.add("Images")
Pkg.add("Finch")

using FileIO
using Images
using Finch
using Base.Filesystem

function get_compression_ratio(filepath::String)
    A = load(filepath)
    m, n = size(A)
    
    B = fill((0, 0, 0), m, n)

    for i in 1:m
        for j in 1:n
            c = A[i, j]
            B[i, j] = (
                round(Int, 255 * Float64(c.r)),
                round(Int, 255 * Float64(c.g)),
                round(Int, 255 * Float64(c.b))
            )
        end
    end

    T = Tensor(Dense(SparseRunList(Element((0,0,0)))), B)

    original_entries = m * n

    compressed_entries = T.lvl.lvl.ptr[end] - 1

    compressed_entries = max(1, compressed_entries)

    return original_entries / compressed_entries
end


function extract_highly_compressed_images(src_dirs, dest_dir; threshold=10.0, top_n=5)
    mkpath(dest_dir)

    candidates = Vector{Tuple{Float64,String,String}}()

    for dir in src_dirs
        if !isdir(dir)
            println("Warning: Directory '$dir' not found. Skipping.")
            continue
        end

        for filename in readdir(dir)
            if endswith(lowercase(filename), ".jpg")
                filepath = joinpath(dir, filename)

                try
                    ratio = get_compression_ratio(filepath)
                    if ratio >= threshold
                        push!(candidates, (ratio, filepath, filename))
                    else
                        println("Skipping $filename (Compression: $(round(ratio, digits=2))x)")
                    end
                catch e
                    println("Error processing $filepath: $e")
                end
            end
        end
    end

    sort!(candidates, by=x->x[1], rev=true)
    selected = first(candidates, min(top_n, length(candidates)))

    for (ratio, filepath, filename) in selected
        destpath = joinpath(dest_dir, filename)
        println("Moving $filename (Compression: $(round(ratio, digits=2))x) to $dest_dir")
        try
            mv(filepath, destpath; force=true)
        catch e
            println("Error moving $filepath to $destpath: $e")
        end
    end

    println("\nProcess complete! Top $(length(selected)) highly compressed files moved to: $dest_dir")
end

function select_best_rle_image(dir::String)
    best_path = ""
    best_ratio = 0.0

    for filename in readdir(dir)
        if endswith(lowercase(filename), ".jpg")
            filepath = joinpath(dir, filename)
            try
                ratio = get_compression_ratio(filepath)
                if ratio > best_ratio
                    best_ratio = ratio
                    best_path = filepath
                end
            catch e
                println("Failed on $filepath: $e")
            end
        end
    end

    return best_path, best_ratio
end

best_file, best_ratio = select_best_rle_image("highly_compressed")
println("Best RLE image: $best_file (ratio=$(round(best_ratio, digits=2))x)")

extract_highly_compressed_images(["highly_compressed"], "very_highly_compressed", threshold=10.0, top_n=5)

# source_directories = ["music", "machinery", "sport", "tourism",]
# destination_directory = "highly_compressed"

# extract_highly_compressed_images(source_directories, destination_directory, threshold=10.0)

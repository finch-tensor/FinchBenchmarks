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


function extract_highly_compressed_images(src_dirs, dest_dir; threshold=10.0)
    mkpath(dest_dir) 

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
                        println("Saving $filename (Compression: $(round(ratio, digits=2))x)")
                        cp(filepath, joinpath(dest_dir, filename), force=true)
                    else
                        println("Skipping $filename (Compression: $(round(ratio, digits=2))x)")
                    end
                catch e
                    println("Error processing $filepath: $e")
                end
            end
        end
    end
    
    println("\nProcess complete! Highly compressed files are in: $dest_dir")
end

source_directories = ["music", "machinery"]
destination_directory = "highly_compressed"

extract_highly_compressed_images(source_directories, destination_directory, threshold=10.0)

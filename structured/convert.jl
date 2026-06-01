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

    # --- RAW CALCULATION ---
    # 1. Total pixels in the original matrix
    original_entries = m * n

    # 2. Total runs in the compressed tensor
    # In Finch, a SparseRunList stores a `ptr` array to track the starts/ends of runs 
    # across dimensions. The last value in this array, minus 1, gives the exact 
    # number of non-background runs that were grouped and stored.
    compressed_entries = T.lvl.lvl.ptr[end] - 1

    # Safeguard: If the image is perfectly black (0,0,0 everywhere), Finch stores 0 runs.
    # We enforce a minimum of 1 to avoid a DivideByZero error.
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

A = load("structured/music/3live.ru.jpg")
m, n = size(A)
_A = fill((0, 0, 0), m, n)

for i in 1:m
    for j in 1:n
        c = A[i, j]

        _A[i, j] = (
            round(Int, 255 * Float64(c.r)),
            round(Int, 255 * Float64(c.g)),
            round(Int, 255 * Float64(c.b))
        )
    end
end

T = Tensor(Dense(SparseRunList(Element((0,0,0)))), _A)

dev = cpu(:t, 1)

S = Tensor(Dense(Shard(dev, SparseRunList(Element((0,0,0))))))

@finch begin
    for i = parallel(_, dev), j = _
        S[i, j] = T[i, j]
    end
end


A = load("highly_compressed/benjamin.duboc.free.fr.jpg")
m, n = size(A)
_A = zeros(m, n)

for i in 1:m
    for j in 1:n
        c = A[i, j]

        _A[i, j] = c.r * 0.299 + c.g * 0.587 + c.b * 0.114
    end
end

_A_r = Tensor(Dense(SparseRunList(Element(0.0))), _A)

B = load("highly_compressed/guitars.com.jpg")
_B = zeros(m, n)

for i in 1:m
    for j in 1:n
        c = B[i, j]

        _B[i, j] = c.r * 0.299 + c.g * 0.587 + c.b * 0.114
    end
end

_B_r = Tensor(Dense(SparseRunList(Element(0.0))), _B)

dev = cpu(:t, 2)

S = Tensor(Dense(Shard(dev, SparseRunList(Element(0.0)))))

@finch begin
    S .= 0.0
    for j = parallel(_, dev), i = _
        S[i, j] = _A_r[i, j] + _B_r[i, j]
    end
end


n_S = Tensor(Dense(SparseRunList(Element(0.0))), m, n)

@finch begin
    for i = _, j = _
        n_S[i, j] = _A_r[i, j] + _B_r[i, j]
    end
end
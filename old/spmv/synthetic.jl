using Random
using LinearAlgebra

function random_permutation_matrix(n)
    perm = randperm(n) 
    return sparse(collect(1:n), perm, ones(n))
end

function reverse_permutation_matrix(n)
    perm = reverse(collect(1:n))
    return sparse(collect(1:n), perm, ones(n))
end

function banded_matrix(n, b)
    banded = zeros(n, n)
    for i in 1:n
        for j in max(1, i - b):min(n, i + b)
            banded[i, j] = j - i + 1 
        end
    end
    return banded
end

function upper_triangle_matrix(n)
    tri = zeros(n, n)
    for i in 1:n
        for j in i:n
            tri[i, j] = rand()
        end
    end
    return tri
end
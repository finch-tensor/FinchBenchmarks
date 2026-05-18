erode_opencv_kernel(data, filter, niters) = OpenCV.erode(data, filter, iterations=niters)

function erode_opencv((img, niters),)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter, $niters) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter, niters)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

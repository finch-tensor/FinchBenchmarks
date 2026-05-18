function hist_opencv((img, mask),)
    # step 1: gray scale the image
    # vector of images - don't ask
    imgs = Vector{OpenCV.InputArray}([reshape(img, 1, size(img)...)])
    mask = reshape(mask, 1, size(mask)...)
    result = Ref{Any}()
    channels = Vector{Int32}([0])
    bins = Vector{Int32}([16])
    regions = Vector{Float32}([0, 256])
    time = @belapsed OpenCV.calcHist($imgs, $channels, $mask, $bins, $regions)
    result = OpenCV.calcHist(imgs, channels, mask, bins, regions)
    (;time=time, output=map(x->round(Int, x), reshape(Array(result), :)), mem = summarysize(img), nnz = length(img))
end

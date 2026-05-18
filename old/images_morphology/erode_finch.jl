for (input, output, tmp) in [
    [
        Tensor(Dense(Dense(Element(UInt(0))))),
        Tensor(Dense(Dense(Element(UInt(0))))),
        Tensor(Dense(Element(UInt(0))))
    ],
]
    eval(Finch.@finch_kernel function erode_finch_bits_kernel(output, input, tmp)
        output .= 0
        for y = _
            tmp .= 0
            for x = _
                tmp[x] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
            end
            for x = _
                let tl = coalesce(tmp[~(x-1)], ~(UInt(0))), t = tmp[x], tr = coalesce(tmp[~(x+1)], ~(UInt(0)))
                    output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                end
            end
        end
        return output
    end)
end

for (input, output, tmp, mask) in [
    [
        Tensor(Dense(Dense(Element(UInt(0))))),
        Tensor(Dense(Dense(Element(UInt(0))))),
        Tensor(Dense(Element(UInt(0)))),
        Tensor(Dense(SparseList(Pattern()))),
    ],
]
    eval(Finch.@finch_kernel function erode_finch_bits_mask_kernel(output, input, tmp, mask)
        output .= 0
        for y = _
            tmp .= 0
            for x = _
                if mask[x, y]
                    tmp[x] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
                end
            end
            for x = _
                if mask[x, y]
                    let tl = coalesce(tmp[~(x-1)], ~(UInt(0))), t = tmp[x], tr = coalesce(tmp[~(x+1)], ~(UInt(0)))
                        let res = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                            output[x, y] = res
                        end
                    end
                end
            end
        end
        return (output)
    end)
end

for (input, output, tmp) in [
    [
        Tensor(Dense(Dense(Element(false)))),
        Tensor(Dense(Dense(Element(false)))),
        Tensor(Dense(Element(false))),
    ],
    [
        Tensor(Dense(SparseRLE(Pattern())))
        Tensor(Dense(SparseRLE(Pattern())))
        Tensor(SparseRLE(Pattern(), merge=false))
    ],
]
    eval(Finch.@finch_kernel function erode_finch_kernel(output, input, tmp)
        output .= false
        for y = _
            tmp .= false
            for x = _
                tmp[x] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
            end
            for x = _
                output[x, y] = coalesce(tmp[~(x-1)], true) & tmp[x] & coalesce(tmp[~(x+1)], true)
            end
        end
        return output
    end)
end

erode_finch_kernel2(output, input, tmp, niters) = begin
    (output, input) = (input, output)
    for i in 1:niters
        (output, input) = (input, output)
        output = erode_finch_kernel(output, input, tmp).output
    end
    return output
end

function erode_finch((img, niters),)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel2($output, $input, $tmp, $niters) evals=1
    input = Tensor(Dense(Dense(Element(false))), img)
    output = erode_finch_kernel2(output, input, tmp, niters)
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

erode_finch_bits_kernel2(output, input, tmp, niters) = begin
    (output, input) = (input, output)
    for i in 1:niters
        (output, input) = (input, output)
        output = erode_finch_bits_kernel(output, input, tmp).output
    end
    return output
end

function erode_finch_bits((img, niters),)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel2($outputb, $inputb, $tmpb, $niters) evals=1
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = erode_finch_bits_kernel2(outputb, inputb, tmpb, niters)
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_rle((img, niters),)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed erode_finch_kernel2($output, $input, $tmp, $niters) evals=1
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = erode_finch_kernel2(output, input, tmp, niters)
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

erode_finch_bits_mask_kernel2(output, input, tmp, niters) = begin
    i = 0
    while i < niters
        mask = Tensor(Dense(SparseList(Pattern())))
        @finch begin
            mask .= false
            for j = _, i = _
                if input[i, j] != 0
                    mask[i, j] = true
                end
            end
        end
        for _ = 1:8
            output = erode_finch_bits_mask_kernel(output, input, tmp, mask).output
            (output, input) = (input, output)
            i += 1
            if i == niters
                break
            end
        end
    end
    return input
end

function erode_finch_bits_mask((img, niters),)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    #maskb = Tensor(Dense(SparseList(Pattern())), imgb .!= 0)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_mask_kernel2($outputb, $inputb, $tmpb, $niters) evals=1
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = erode_finch_bits_mask_kernel2(outputb, inputb, tmpb, niters)
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end
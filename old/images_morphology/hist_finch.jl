for (bins, img, mask) in [
    [
        Tensor(Dense(Element(0)))
        Tensor(Dense(Dense(Element(UInt8(0)))))
        Tensor(Dense(Dense(Element(false))))
    ],
    [
        Tensor(Dense(Element(0)))
        Tensor(Dense(Dense(Element(UInt8(0)))))
        Tensor(Dense(SparseRLE(Pattern())))
    ],
]
    eval(@finch_kernel function hist_finch_kernel(bins, img, mask)
        bins .= 0 
        for x=_
            for y=_
                if mask[y, x]
                    bins[div(img[y, x], 16) + 1] += 1
                end
            end
        end
        return bins
    end)
end

function hist_finch((img, mask),)
    bins = Tensor(Dense(Element(0)), undef, 16)
    img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
    mask = Tensor(Dense(Dense(Element(false))), mask)
    time = @belapsed hist_finch_kernel($bins, $img, $mask)
    result = hist_finch_kernel(bins, img, mask)
    (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
end

function hist_finch_rle((img, mask),)
    bins = Tensor(Dense(Element(0)), undef, 16)
    img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
    mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
    time = @belapsed hist_finch_kernel($bins, $img, $mask)
    result = hist_finch_kernel(bins, img, mask)
    (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
end

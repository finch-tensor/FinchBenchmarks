using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
#unicodeplots()
pyplot()

include("plot_labels.jl")

data = DataFrame(open("all_pairs_results.json", "r") do f
    JSON.parse(f)
end)

interest = [
    "finch_sparse",
    "finch_gallop",
    "finch_vbl",
    "finch_rle",
]

ref = data[isequal("opencv").(data.method), [:matrix, :time]]
rename!(ref, :time => :ref)
data = outerjoin(data, ref, on = [:matrix])
data.speedup = data.ref ./ data.time

data = data[map(m -> m in interest, data.method), :]
group = CategoricalArray(label.(data.method), levels=label.(interest))
matrix = CategoricalArray(label.(data.matrix), levels=label.(["mnist_train", "emnist_letters_train", "emnist_digits_train", "omniglot_train"]))

p = groupedbar(data.matrix,
    data.speedup,
    group=group,
    xlabel="Dataset",
    ylabel = "Speedup Over OpenCV"
)

savefig(p, "all_pairs.png")
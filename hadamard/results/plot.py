import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

kernel = sys.argv[1]

input_file = f"./hadamard_{kernel}_results.json"
output_file = f"./hadamard_{kernel}_speedup.png"

ktype = ""
if kernel == "mirror":
    ktype = "Identity "

with open(input_file) as f:
    raw = json.load(f)

RENAME = {
    "shard_implementation": "wingspan",
    "eigen_impl":           "eigen",
    "graphblas_impl":       "graphblas",
}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    if method not in RENAME:
        continue
    data.setdefault(m, {})[RENAME[method]] = entry["time"]

matrices = list(data.keys())

methods = ["wingspan", "eigen", "graphblas"]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    mkl_t = data[mat]["graphblas"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(mkl_t / t if t else float("nan"))

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.18
group_gap  = 0.06
x = np.arange(n_matrices)

COLORS = {
    "wingspan": "#E69F00",
    "mkl":      "#009E73",
    "eigen":    "#CC79A7",
    "graphblas": "#56B4E9",
}

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, meth in enumerate(methods):
    offsets = x + (i - n_methods / 2 + 0.5) * (bar_width + group_gap / n_methods)
    ax.bar(
        offsets,
        speedups[meth],
        width=bar_width,
        label=meth,
        color=COLORS[meth],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--", zorder=2)


ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_xticks(x)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_xticklabels(matrices, rotation=15, ha="right")
ax.set_ylabel("Speedup")
ax.set_title(f"{ktype}Hadamard Speedup Results", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig(output_file, dpi=200)
print("Saved.")
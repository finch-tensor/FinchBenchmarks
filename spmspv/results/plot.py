import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "./spmspv_wingspan.json"
birdseed_file = sys.argv[2] if len(sys.argv) > 2 else "./spmspv_birdseed.json"
with open(input_file) as f:
    raw = json.load(f)

try:
    with open(birdseed_file) as f:
        raw += json.load(f)
except FileNotFoundError:
    print(f"warning: {birdseed_file} not found, plotting without birdseed")

# "<series>-static" and "<series>-dynamic" fold into one series, keeping the faster.
def series(method):
    for suffix in ("-static", "-dynamic"):
        if method.endswith(suffix):
            return method[: -len(suffix)]
    return method

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    meth = series(entry["method"])
    t = entry["time"]
    current = data.setdefault(m, {}).get(meth)
    data[m][meth] = t if current is None else min(current, t)

matrices = list(data.keys())

# eigen is the baseline at 1.0; drop series with no results at all
methods = [
    "wingspan-bytemap", "wingspan-hash",
    "birdseed-bytemap", "birdseed-hash",
    "graphblas", "eigen",
]
methods = [meth for meth in methods if any(meth in data[mat] for mat in matrices)]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    eigen_t = data[mat]["eigen"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(eigen_t / t if t else float("nan"))

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.8 / n_methods
group_gap  = 0.06
x = np.arange(n_matrices)

COLORS = {
    "wingspan-bytemap":  "#E69F00",
    "wingspan-hash":     "#F0E442",
    "birdseed-bytemap":  "#D55E00",
    "birdseed-hash":     "#009E73",
    "graphblas":         "#56B4E9",
    "eigen":             "#CC79A7",
}

fig, ax = plt.subplots(figsize=(14, 6))
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
ax.set_title("SpMSpV Speedup Results", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./spmspv_speedup.png", dpi=200)
print("Saved.")

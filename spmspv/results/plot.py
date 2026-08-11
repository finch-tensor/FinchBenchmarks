import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open("./spmspv_results.json") as f:
    raw = json.load(f)

WINGSPAN_METHODS = {"coalesce_static", "coalesce_dynamic"}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    t = entry["time"]
    if method in WINGSPAN_METHODS:
        current = data.setdefault(m, {}).get("wingspan")
        data[m]["wingspan"] = t if current is None else min(current, t)
    else:
        data.setdefault(m, {})[method] = t

matrices = list(data.keys())

# Order: graphblas is baseline at 1.0
methods = ["wingspan", "graphblas", "eigen"]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    graphblas_t = data[mat]["eigen"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(graphblas_t / t if t else float("nan"))

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.18
group_gap  = 0.06
x = np.arange(n_matrices)

COLORS = {
    "wingspan":  "#E69F00", 
    "graphblas":       "#56B4E9", 
    "eigen":           "#CC79A7",
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
ax.set_title("SpMSpV Speedup Results", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./spmspv_speedup.png", dpi=200)
print("Saved.")

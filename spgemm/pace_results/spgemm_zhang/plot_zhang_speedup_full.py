import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────

with open("./zhang_small_results_full.json") as f:
    raw = json.load(f)

RENAME = {
    "spgemm_finch_outer":             "wingspan_outer",
    "spgemm_finch_inner":             "wingspan_inner",
    "spgemm_finch_gustavson_dynamic": "wingspan_gustavson",
    "spgemm_finch_nested":            "wingspan_nested",
    "spgemm_eigen":                   "eigen",
    "spgemm_mkl":                     "mkl",
    "spgemm_graphblas":               "graphblas",
}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    if method not in RENAME:
        continue
    data.setdefault(m, {})[RENAME[method]] = entry["time"]

matrices = list(data.keys())

methods = [
    "wingspan_gustavson",
    "mkl",
    "eigen",
    "graphblas",
    "wingspan_outer",
    "wingspan_inner",
    "wingspan_nested",
]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    mkl_t = data[mat]["mkl"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(mkl_t / t if t else float("nan"))

# ── Layout ──────────────────────────────────────────────────────────────────

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.1
group_gap  = 0.08
x = np.arange(n_matrices)

# Wong (2011) colorblind-safe palette — distinguishable under
# deuteranopia, protanopia, and tritanopia
COLORS = {
    "wingspan_gustavson": "#E69F00",  # orange
    "mkl":                "#56B4E9",  # sky blue
    "eigen":              "#009E73",  # bluish green
    "graphblas":          "#D55E00",  # vermillion
    "wingspan_outer":     "#F0E442",  # yellow
    "wingspan_inner":     "#0072B2",  # blue
    "wingspan_nested":    "#CC79A7",  # reddish purple
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

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=15, ha="right")
ax.set_ylabel("Speedup")
ax.set_title("Speedup on Small Zhang Matrices", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper right", framealpha=0.85, edgecolor="#cccccc", ncol=3)

plt.tight_layout()
plt.savefig("./zhang_speedup_full.png", dpi=150, bbox_inches="tight")
print("Saved.")
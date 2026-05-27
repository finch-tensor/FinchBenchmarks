import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────

with open("./HB_sum_results.json") as f:
    raw = json.load(f)

RENAME = {
    "coalesce_impl": "wingspan",
    "taco_impl":     "taco",
    "eigen_impl":    "eigen",
    "mkl_impl":      "mkl",
}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    if method not in RENAME:
        continue
    data.setdefault(m, {})[RENAME[method]] = entry["time"]

matrices = list(data.keys())

# Order: wingspan and taco adjacent (taco is baseline at 1.0)
methods = ["wingspan", "taco", "mkl", "eigen"]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    taco_t = data[mat]["taco"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(taco_t / t if t else float("nan"))

# ── Layout ──────────────────────────────────────────────────────────────────

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.18
group_gap  = 0.06
x = np.arange(n_matrices)

# Wong (2011) colorblind-safe palette
COLORS = {
    "wingspan": "#E69F00",  # orange
    "taco":     "#56B4E9",  # sky blue
    "mkl":      "#009E73",  # bluish green
    "eigen":    "#CC79A7",  # reddish purple
}

fig, ax = plt.subplots(figsize=(16, 6))
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

# Label eigen bars with their speedup value
eigen_idx = methods.index("eigen")
for j, mat in enumerate(matrices):
    val = speedups["eigen"][j]
    if not np.isnan(val):
        bar_x = x[j] + (eigen_idx - n_methods / 2 + 0.5) * (bar_width + group_gap / n_methods)
        label = "<0.01×" if val < 0.01 else f"{val:.2f}×"
        ax.text(
            bar_x, val + 0.01,
            label,
            ha="left", va="bottom",
            fontsize=7.5, color=COLORS["eigen"],
            rotation=90,
            rotation_mode="anchor",
        )

ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Speedup", fontsize=11)
ax.set_title("Vector Sum Reduction Results", fontsize=13, fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./vector_sum_speedup.png", dpi=150, bbox_inches="tight")
print("Saved.")
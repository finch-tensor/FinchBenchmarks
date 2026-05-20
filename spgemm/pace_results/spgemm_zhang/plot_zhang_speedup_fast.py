import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────

with open("./zhang_large_results_fast.json") as f:
    raw = json.load(f)

RENAME = {
    "spgemm_finch_gustavson_dynamic": "wingspan_gustavson_dynamic",
    "spgemm_finch_gustavson_static":  "wingspan_gustavson_static",
    "spgemm_eigen":                   "eigen",
    "spgemm_mkl":                     "mkl",
}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    if method not in RENAME:
        continue
    data.setdefault(m, {})[RENAME[method]] = entry["time"]

# Pick best gustavson (lower runtime = higher speedup)
for m in data:
    dyn = data[m].get("wingspan_gustavson_dynamic")
    sta = data[m].get("wingspan_gustavson_static")
    if dyn is not None and sta is not None:
        data[m]["wingspan_gustavson"] = min(dyn, sta)
    elif dyn is not None:
        data[m]["wingspan_gustavson"] = dyn
    else:
        data[m]["wingspan_gustavson"] = sta

matrices = list(data.keys())
methods  = ["wingspan_gustavson", "mkl", "eigen"]

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    mkl_t = data[mat]["mkl"]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(mkl_t / t if t else float("nan"))

# ── Layout ──────────────────────────────────────────────────────────────────

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.22
group_gap  = 0.06
x = np.arange(n_matrices)

# Wong (2011) colorblind-safe palette
COLORS = {
    "wingspan_gustavson": "#E69F00",  # orange
    "mkl":                "#56B4E9",  # sky blue
    "eigen":              "#009E73",  # bluish green
}

fig, ax = plt.subplots(figsize=(18, 6))
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
ax.set_xticklabels(matrices, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Speedup", fontsize=11)
ax.set_title("Speedup on Large Zhang Matrices", fontsize=13, fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./zhang_large_speedup.png", dpi=150, bbox_inches="tight")
print("Saved.")

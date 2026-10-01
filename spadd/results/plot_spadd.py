import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

# usage: plot_spadd.py <kernel> [results.json] [birdseed_results.json] [nacho_results.json]
kernel = sys.argv[1]
input_file = sys.argv[2] if len(sys.argv) > 2 else f"./spadd_{kernel}_results.json"
birdseed_file = sys.argv[3] if len(sys.argv) > 3 else f"./spadd_birdseed_{kernel}.json"
nacho_file = sys.argv[4] if len(sys.argv) > 4 else f"./spadd_nacho_{kernel}.json"
output_file = f"./spadd_{kernel}_speedup.png"
ktype = ""
if kernel == "mirror":
    ktype = "Identity "
with open(input_file) as f:
    raw = json.load(f)

# birdseed is run with diff julia env
try:
    with open(birdseed_file) as f:
        birdseed_rows = [e for e in json.load(f) if e["method"] == "birdseed_spadd"]
except FileNotFoundError:
    print(f"warning: {birdseed_file} not found, plotting without birdseed")
    birdseed_rows = []

# nacho is run by its own script (run_spadd_nacho.sh)
try:
    with open(nacho_file) as f:
        nacho_rows = [e for e in json.load(f) if e["method"] == "nacho_dcsr"]
except FileNotFoundError:
    print(f"warning: {nacho_file} not found, plotting without nacho")
    nacho_rows = []

RENAME = {
    "wingspan_spadd": "wingspan",
    "mkl_impl":       "mkl",
    "eigen_impl":     "eigen",
    "graphblas_impl": "graphblas",
}

data: dict = {}
for entry in raw:
    m = entry["matrix"].split("/")[-1]
    method = entry["method"]
    if method not in RENAME:
        continue
    data.setdefault(m, {})[RENAME[method]] = entry["time"]
for entry in birdseed_rows:
    m = entry["matrix"].split("/")[-1]
    data.setdefault(m, {})["birdseed"] = entry["time"]
for entry in nacho_rows:
    m = entry["matrix"].split("/")[-1]
    data.setdefault(m, {})["nacho"] = entry["time"]

matrices = list(data.keys())

# Order: mkl is baseline at 1.0
methods = ["wingspan", "birdseed", "nacho", "mkl", "eigen", "graphblas"]
if not birdseed_rows:
    methods.remove("birdseed")
if not nacho_rows:
    methods.remove("nacho")

speedups: dict = {m: [] for m in methods}
for mat in matrices:
    mkl_t = data[mat].get("mkl", float("nan"))
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(mkl_t / t if t else float("nan"))

n_matrices = len(matrices)
n_methods  = len(methods)
bar_width  = 0.8 / n_methods
group_gap  = 0.06
x = np.arange(n_matrices)

COLORS = {
    "wingspan": "#E69F00",
    "birdseed":     "#D55E00",
    "nacho":    "#0072B2",
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
ax.set_title(f"{ktype}SpAdd Speedup Results", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig(output_file, dpi=200)
print("Saved.")

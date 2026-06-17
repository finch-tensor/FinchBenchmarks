import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = "./dot_results.json"

with open(DATA_PATH) as f:
    raw = json.load(f)

# --- Rename methods ---
METHOD_NAMES = {
    "coalesce_impl":  "wingspan",
    "eigen_impl":     "eigen",
    "graphblas_impl": "graphblas",
}

# --- Collect times keyed by (matrix, method) ---
times = {}
for entry in raw:
    matrix = entry["matrix"].replace("HB/", "")
    method = METHOD_NAMES[entry["method"]]
    times[(matrix, method)] = entry["time"]

methods_required = {"wingspan", "eigen", "graphblas"}
matrices = sorted({m for (m, _) in times if all((m, meth) in times for meth in methods_required)})

# --- Compute speedups: graphblas_time / method_time ---
graphblas_times = {m: times[(m, "graphblas")] for m in matrices}

wingspan_speedup  = [graphblas_times[m] / times[(m, "wingspan")]  for m in matrices]
eigen_speedup     = [graphblas_times[m] / times[(m, "eigen")]     for m in matrices]
graphblas_speedup = [1.0] * len(matrices)

# --- Wong color palette ---
COLOR_WINGSPAN  = "#E69F00"
COLOR_EIGEN     = "#56B4E9"
COLOR_GRAPHBLAS = "#999999"
COLOR_DASHLINE  = "#CC79A7"

# --- Plot ---
x = np.arange(len(matrices))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))

bars1 = ax.bar(x - width, wingspan_speedup,  width, label="Wingspan",  color=COLOR_WINGSPAN)
bars2 = ax.bar(x,         eigen_speedup,     width, label="Eigen",     color=COLOR_EIGEN)
bars3 = ax.bar(x + width, graphblas_speedup, width, label="GraphBLAS", color=COLOR_GRAPHBLAS)

# Dashed reference line at 1x
ax.axhline(1.0, color=COLOR_DASHLINE, linewidth=1.5, linestyle="--", label="1× baseline")

ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Speedup", fontsize=11)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}×"))

ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.title("Sparse Vector Dot Product Results")

fig.tight_layout()
fig.savefig("./dot_speedup.png", bbox_inches="tight", dpi=150)
plt.show()

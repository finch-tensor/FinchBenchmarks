"""
Speedup bar chart for SpGEMM benchmarks.

Produces a single figure with two subplots:
  Left  – zhang_small_balance_results_chk.json  (small matrices)
  Right – zhang_large_balance_results_chk.json  (large matrices)

Baseline: spgemm_mkl (shown at 1× on every matrix)
Speedup  = T_baseline / T_method

WingSpan is shown as a single bar using min(dynamic, static) time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE = "spgemm_mkl"

# Internal method keys used for loading raw times
RAW_METHODS = [
    "spgemm_eigen",
    "spgemm_mkl",
    "spgemm_finch_gustavson_dynamic",
    "spgemm_finch_gustavson_static",
]

# Methods as presented in the chart (WingSpan merged into one)
PLOT_METHODS = [
    "spgemm_eigen",
    "spgemm_mkl",
    "spgemm_finch_wingspan",   # synthetic key for min(dynamic, static)
]

METHOD_LABELS = {
    "spgemm_eigen":            "Eigen",
    "spgemm_mkl":              "MKL",
    "spgemm_finch_wingspan":   "WingSpan",
}

METHOD_COLORS = {
    "spgemm_eigen":            "#4477AA",
    "spgemm_mkl":              "#EE6677",
    "spgemm_finch_wingspan":   "#CCBB44",
}

DATASETS = [
    ("zhang_small_balance_results_chk.json", "Small Matrices"),
    ("zhang_large_balance_results_chk.json", "Large Matrices"),
]

BAR_WIDTH = 0.2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_speedup(path: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Return (matrices_sorted_by_baseline_time, speedup_dict).

    WingSpan time = min(dynamic, static); speedup relative to MKL.
    """
    with open(path) as f:
        records = json.load(f)

    times: dict[str, dict[str, float]] = {}
    for r in records:
        mat = r["matrix"].split("/")[-1]
        times.setdefault(mat, {})[r["method"]] = r["time"]

    # Merge the two Finch variants into a single WingSpan entry
    for mat in times:
        dyn = times[mat].get("spgemm_finch_gustavson_dynamic")
        sta = times[mat].get("spgemm_finch_gustavson_static")
        if dyn is not None and sta is not None:
            times[mat]["spgemm_finch_wingspan"] = min(dyn, sta)
        elif dyn is not None:
            times[mat]["spgemm_finch_wingspan"] = dyn
        elif sta is not None:
            times[mat]["spgemm_finch_wingspan"] = sta

    # Keep only matrices that have all plotted methods
    matrices = sorted(
        [m for m in times if all(meth in times[m] for meth in PLOT_METHODS)],
        key=lambda m: times[m][BASELINE],
    )

    speedup = {
        mat: {
            method: times[mat][BASELINE] / times[mat][method]
            for method in PLOT_METHODS
        }
        for mat in matrices
    }
    return matrices, speedup


def draw_subplot(ax, matrices, speedup, title: str):
    n = len(matrices)
    x = np.arange(n)
    n_methods = len(PLOT_METHODS)
    offsets_base = np.linspace(
        -(n_methods - 1) / 2 * BAR_WIDTH,
         (n_methods - 1) / 2 * BAR_WIDTH,
        n_methods,
    )

    for i, method in enumerate(PLOT_METHODS):
        heights = [speedup[mat][method] for mat in matrices]
        ax.bar(
            x + offsets_base[i],
            heights,
            width=BAR_WIDTH,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="1× (MKL baseline)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Speedup", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_xlim(-0.5, n - 0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=False)

for ax, (path, title) in zip(axes, DATASETS):
    matrices, speedup = load_speedup(path)
    draw_subplot(ax, matrices, speedup, title)

fig.suptitle("SpGEMM Speedup Comparison", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("spgemm_speedup.png", dpi=150, bbox_inches="tight")
print("Saved spgemm_speedup.png")

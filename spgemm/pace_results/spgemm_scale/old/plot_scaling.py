"""
Combined strong- and weak-scaling execution-time plot for SpGEMM benchmarks.

File naming convention: spgemm_scale_threads_<N>.json  (N = 1, 2, 4, 8, 16)

Strong scaling: fixed matrix rand_16384, vary thread count.
Weak scaling:   thread count paired with proportionally larger matrix
    1  → rand_1024
    2  → rand_2048
    4  → rand_4096
    8  → rand_8192
   16  → rand_16384
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THREAD_COUNTS = [1, 2, 4, 8, 16]

# Strong scaling: one fixed matrix
STRONG_MATRIX = "rand_16384"

# Weak scaling: thread → matrix
SIZE_FOR_THREAD = {
    1:  "rand_1024",
    2:  "rand_2048",
    4:  "rand_4096",
    8:  "rand_8192",
    16: "rand_16384",
}

METHODS = [
    "spgemm_eigen",
    "spgemm_mkl",
    "spgemm_finch_gustavson",
]

METHOD_LABELS = {
    "spgemm_eigen":           "Eigen",
    "spgemm_mkl":             "MKL",
    "spgemm_finch_gustavson": "WingSpan (Gustavson)",
}

METHOD_COLORS = {
    "spgemm_eigen":           "#4477AA",
    "spgemm_mkl":             "#EE6677",
    "spgemm_finch_gustavson": "#228833",
}

METHOD_MARKERS = {
    "spgemm_eigen":           "o",
    "spgemm_mkl":             "s",
    "spgemm_finch_gustavson": "^",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_file(n_threads: int) -> list[dict]:
    fname = f"spgemm_scale_threads_{n_threads}.json"
    with open(fname) as f:
        return json.load(f)


def extract_time(records: list[dict], matrix_key: str, method: str) -> float:
    for r in records:
        mat = r["matrix"].split("/")[-1].replace(".ttx", "")
        if mat == matrix_key and r["method"] == method:
            return r["time"]
    raise KeyError(f"No record found for matrix={matrix_key}, method={method}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Cache each file so we only read it once
_cache: dict[int, list[dict]] = {}

def get_records(n: int) -> list[dict]:
    if n not in _cache:
        _cache[n] = load_file(n)
    return _cache[n]


# Strong-scaling times: fixed matrix, varying threads
strong_times: dict[str, dict[int, float]] = {m: {} for m in METHODS}
for n in THREAD_COUNTS:
    records = get_records(n)
    for method in METHODS:
        strong_times[method][n] = extract_time(records, STRONG_MATRIX, method)

# Weak-scaling times: thread-proportional matrix size
weak_times: dict[str, dict[int, float]] = {m: {} for m in METHODS}
for n in THREAD_COUNTS:
    records = get_records(n)
    mat_key = SIZE_FOR_THREAD[n]
    for method in METHODS:
        weak_times[method][n] = extract_time(records, mat_key, method)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SpGeMM Runtime Scaling",
             fontsize=13, fontweight="bold", y=0.98)

x        = np.array(THREAD_COUNTS)
x_labels = [str(n) for n in THREAD_COUNTS]

def style_ax(ax, ylabel: str, title: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Threads", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)


# --- Left panel: strong scaling ---
ax = axes[0]
for method in METHODS:
    y = [strong_times[method][n] * 1e3 for n in THREAD_COUNTS]
    ax.plot(x, y,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2, markersize=7)
style_ax(ax, "Wall time (ms)", f"Strong Scaling")

# --- Right panel: weak scaling ---
ax = axes[1]
for method in METHODS:
    y = [weak_times[method][n] * 1e3 for n in THREAD_COUNTS]
    ax.plot(x, y,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2, markersize=7)
style_ax(ax, "Wall time (ms)", "Weak Scaling")

plt.tight_layout()
plt.savefig("spgemm_scaling.png", dpi=150, bbox_inches="tight")
print("Saved spgemm_scaling.png")
plt.show()

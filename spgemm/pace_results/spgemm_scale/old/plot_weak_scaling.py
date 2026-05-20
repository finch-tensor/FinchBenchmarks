"""
Weak scaling plot for SpGEMM benchmarks.

File naming convention: spgemm_scale_threads_<N>.json  (N = 1, 2, 4, 8, 16)
Weak-scaling pairing (threads → matrix size):
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

THREAD_COUNTS   = [1, 2, 4, 8, 16]
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
    "spgemm_finch_gustavson": "Finch (Gustavson)",
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
# Load data
# ---------------------------------------------------------------------------

def load_file(n_threads: int) -> list[dict]:
    fname = f"spgemm_scale_threads_{n_threads}.json"
    with open(fname) as f:
        return json.load(f)


def extract_time(records: list[dict], matrix_key: str, method: str) -> float:
    """Return the measured time for a given (matrix, method) pair."""
    for r in records:
        mat = r["matrix"].split("/")[-1].replace(".ttx", "")   # e.g. "rand_1024"
        if mat == matrix_key and r["method"] == method:
            return r["time"]
    raise KeyError(f"No record found for matrix={matrix_key}, method={method}")


# Build time table:  times[method][thread_count] = seconds
times: dict[str, dict[int, float]] = {m: {} for m in METHODS}

for n in THREAD_COUNTS:
    records = load_file(n)
    mat_key = SIZE_FOR_THREAD[n]
    for method in METHODS:
        times[method][n] = extract_time(records, mat_key, method)

# ---------------------------------------------------------------------------
# Compute normalised efficiency  (T_1 / T_N  — ideal = 1.0)
# ---------------------------------------------------------------------------

efficiency: dict[str, dict[int, float]] = {}
for method in METHODS:
    t1 = times[method][1]
    efficiency[method] = {n: t1 / times[method][n] for n in THREAD_COUNTS}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SpGEMM Weak Scaling",
             fontsize=13, fontweight="bold", y=0.97)

x = np.array(THREAD_COUNTS)
x_labels = [str(n) for n in THREAD_COUNTS]

# --- Left panel: raw execution time ---
ax = axes[0]
for method in METHODS:
    y = [times[method][n] * 1e3 for n in THREAD_COUNTS]   # → milliseconds
    ax.plot(x, y,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2, markersize=7)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Threads", fontsize=11)
ax.set_ylabel("Wall time (ms)", fontsize=11)
ax.set_title("Execution Time", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

# --- Right panel: weak-scaling efficiency ---
ax = axes[1]
# Ideal line
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Ideal (efficiency = 1)")

for method in METHODS:
    y = [efficiency[method][n] for n in THREAD_COUNTS]
    ax.plot(x, y,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2, markersize=7)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Threads", fontsize=11)
ax.set_ylabel("Weak-scaling efficiency  (T₁ / Tₙ)", fontsize=11)
ax.set_title("Weak-Scaling Efficiency", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("spgemm_weak_scaling.png", dpi=150, bbox_inches="tight")
print("Saved spgemm_weak_scaling.png")
plt.show()

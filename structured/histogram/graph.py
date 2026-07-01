import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FOLDER = BASE_DIR / "results"
GRAPH_FOLDER = BASE_DIR / "graph"
SPEEDUP_FOLDER = GRAPH_FOLDER / "speedup"

THREADS = 16

RENAME = {
    "coalesce_impl": "wingspan",
    "halide_impl": "halide",
}


def sanitize_matrix_name(matrix: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", matrix).strip("_")


def format_sparsity(x: float) -> str:
    return f"{x:.12f}".rstrip("0").rstrip(".")

def display_matrix_name(matrix: str) -> str:
    if "/" in matrix:
        # Remove first directory
        matrix = matrix.split("/", 1)[1]

        # Remove leading "www."
        matrix = matrix.removeprefix("www.")

        # Remove trailing ".jpg"
        matrix = matrix.removesuffix(".jpg")

    return matrix


def load_results():
    combined = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for path in sorted(RESULTS_FOLDER.glob("hist_*_threads.json")):
        with path.open() as fh:
            records = json.load(fh)

        for entry in records:
            matrix = entry["matrix"]
            if isinstance(matrix, dict):
                matrix = f"{matrix['size']}-{format_sparsity(matrix['sparsity'])}"

            dataset = entry["dataset"]
            method = entry["method"]
            n_threads = entry["n_threads"]

            combined[dataset][matrix][method][n_threads] = entry["time"]

    return combined


def plot_speedup(results, dataset, save_location):
    matrices = sorted(results[dataset].keys())
    labels = [display_matrix_name(m) for m in matrices]

    methods = ["wingspan", "halide"]
    speedups = {m: [] for m in methods}

    for matrix in matrices:
        method_times = {}

        for impl, label in RENAME.items():
            if THREADS in results[dataset][matrix].get(impl, {}):
                method_times[label] = results[dataset][matrix][impl][THREADS]

        if "halide" not in method_times:
            continue

        baseline = method_times["halide"]

        for m in methods:
            t = method_times.get(m)
            speedups[m].append(baseline / t if t else float("nan"))

    n_methods = len(methods)
    n_matrices = len(matrices)

    bar_width = 0.35
    group_gap = 0.06
    x = np.arange(n_matrices)

    COLORS = {
        "wingspan": "#E69F00",
        "halide": "#999999",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, meth in enumerate(methods):
        offsets = x + (i - n_methods / 2 + 0.5) * (
            bar_width + group_gap / n_methods
        )

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
    ax.set_yticks([1])
    ax.set_yticklabels(["1"])
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylabel("Speedup")
    ax.set_title(
        f"Structured Histogram Speedup Results",
        fontweight="bold",
        pad=14,
    )

    ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
    ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(framealpha=0.85, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(save_location, dpi=200)
    plt.close()


if __name__ == "__main__":
    SPEEDUP_FOLDER.mkdir(parents=True, exist_ok=True)

    results = load_results()

    for dataset in sorted(results):
        outfile = SPEEDUP_FOLDER / f"{dataset}_speedup.png"
        plot_speedup(results, dataset, outfile)
        print(f"Saved {outfile}")
import json
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"

# NTHREADS = [i + 1 for i in range(8)]
NTHREADS = [1, 2, 4, 8, 16]

DEFAULT_METHOD = "serial_default_implementation"
METHODS = [
    DEFAULT_METHOD,
    "coalesce_implementation",
]

DATASETS = [
    {"hb_wide": ["HB-bcsstk30", "HB-bcsstk31", "HB-bcsstk32", "HB-cegb2802"]},
]

COLORS = ["gray", "cadetblue", "saddlebrown", "navy", "black"]


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spmspv_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            matrix = (
                result["matrix"][0].replace("/", "-")
                if result["dataset"] != "uniform"
                else f"{result['matrix']['size']}-{result['matrix']['sparsity']}"
            )
            combine_results[result["dataset"]][matrix][result["method"]][
                result["n_threads"]
            ] = result["time"]
    return combine_results


def plot_speedup_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [
                results[dataset][matrix][DEFAULT_METHOD][n_thread]
                / results[dataset][matrix][method][n_thread]
                for n_thread in NTHREADS
            ],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(
        f"SpAdd - Speedup for {dataset}: {matrix} (with respect to {DEFAULT_METHOD})"
    )
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads (log)")
    plt.ylabel(f"Speedup")
    plt.xscale('log', base=2)

    plt.legend()
    plt.savefig(save_location)


def plot_runtime_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [results[dataset][matrix][method][n_thread] for n_thread in NTHREADS],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"SpAdd - Runtime for {dataset}: {matrix}")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads (log)")
    plt.ylabel(f"Runtime (in seconds)")
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)

    plt.legend()
    plt.savefig(save_location)


if __name__ == "__main__":
    results = load_json()
    for datasets in DATASETS:
        for dataset, matrices in datasets.items():
            for matrix in matrices:
                plot_speedup_result(
                    results,
                    dataset,
                    matrix,
                    f"{GRAPH_FOLDER}/{SPEEDUP_FOLDER}/{dataset}-{matrix}.png",
                )
                plot_runtime_result(
                    results,
                    dataset,
                    matrix,
                    f"{GRAPH_FOLDER}/{RUNTIME_FOLDER}/{dataset}-{matrix}.png",
                )

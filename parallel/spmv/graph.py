import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"
MEAN_SPEEDUP_FOLDER = "mean-speedup"

NTHREADS = [i + 1 for i in range(12)]

DEFAULT_METHOD = "serial_default_implementation"
METHODS = [
    # DEFAULT_METHOD,
    # "finch_parallel",
    "static_rows_equal",
    "dynamic_rows_grain_1",
    "dynamic_rows_grain_10",
    "merge",
    # "graph_partition_reorder_merge",
    "graph_partition_weighted_reorder_merge",
]

DATASETS = {
    "uniform": ["1024-0.1", "8192-0.1", "1048576-3000000"],
    "FEMLAB": ["FEMLAB-poisson3Da", "FEMLAB-poisson3Db"],
    "vanHeukelum": [
        "vanHeukelum-cage10",
        # "vanHeukelum-cage11",
        # "vanHeukelum-cage12"
    ],
    "Williams": ["Williams-webbase-1M"],
}
NUM_MATRICES = sum([len(matrices) for matrices in DATASETS.values()])

COLORS = [
    "gray",
    "cadetblue",
    "saddlebrown",
    "navy",
    "black",
    "orange",
    "green",
    "red",
    "purple",
]


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spmv_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            matrix = (
                result["matrix"].replace("/", "-")
                if result["dataset"] != "uniform"
                else f"{result['matrix']['size']}-{result['matrix']['sparsity']}"
            )
            combine_results[result["dataset"]][matrix][result["method"]][
                result["n_threads"]
            ] = result["time"]

    return combine_results


def plot_speedup_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 10))
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

    plt.title(f"Speedup for {dataset}: {matrix} (with respect to {DEFAULT_METHOD})")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


def plot_runtime_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 10))
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

    plt.title(f"Runtime for {dataset}: {matrix}")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Runtime (in seconds)")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


def plot_mean_speedup_result(results, save_location):
    plt.figure(figsize=(10, 10))
    for method, color in zip(METHODS, COLORS):
        speedups = [1] * len(NTHREADS)
        for dataset, matrices in DATASETS.items():
            for matrix in matrices:
                for i, n_thread in enumerate(NTHREADS):
                    speedups[i] *= (
                        results[dataset][matrix][DEFAULT_METHOD][n_thread]
                        / results[dataset][matrix][method][n_thread]
                    )

        mean_speedups = [speedup ** (1 / NUM_MATRICES) for speedup in speedups]
        plt.plot(
            NTHREADS,
            mean_speedups,
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Geometric Mean Speedup (with respect to {DEFAULT_METHOD})")
    # plt.yscale("log", base=10)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


def plot_mean_speedup_separate_result(results, save_folder):
    for method, color in zip(METHODS, COLORS):
        plt.figure(figsize=(10, 10))
        speedups = [1] * len(NTHREADS)
        for dataset, matrices in DATASETS.items():
            for matrix in matrices:
                for i, n_thread in enumerate(NTHREADS):
                    speedups[i] *= (
                        results[dataset][matrix][DEFAULT_METHOD][n_thread]
                        / results[dataset][matrix][method][n_thread]
                    )

        mean_speedups = [speedup ** (1 / NUM_MATRICES) for speedup in speedups]
        plt.plot(
            NTHREADS,
            mean_speedups,
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

        plt.title(
            f"Geometric Mean Speedup for {method} (with respect to {DEFAULT_METHOD})"
        )
        # plt.yscale("log", base=10)
        plt.xticks(NTHREADS)
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup")

        plt.legend()
        plt.savefig(os.path.join(save_folder, f"{method}-mean-speedup.png"))
        plt.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(GRAPH_FOLDER, SPEEDUP_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(GRAPH_FOLDER, RUNTIME_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(GRAPH_FOLDER, MEAN_SPEEDUP_FOLDER), exist_ok=True)

    results = load_json()
    for dataset, matrices in DATASETS.items():
        for matrix in matrices:
            plot_speedup_result(
                results,
                dataset,
                matrix,
                os.path.join(GRAPH_FOLDER, SPEEDUP_FOLDER, f"{dataset}-{matrix}.png"),
            )
            plot_runtime_result(
                results,
                dataset,
                matrix,
                os.path.join(GRAPH_FOLDER, RUNTIME_FOLDER, f"{dataset}-{matrix}.png"),
            )

    plot_mean_speedup_result(results, os.path.join(GRAPH_FOLDER, "mean-speedup.png"))

    plot_mean_speedup_separate_result(
        results, os.path.join(GRAPH_FOLDER, MEAN_SPEEDUP_FOLDER)
    )

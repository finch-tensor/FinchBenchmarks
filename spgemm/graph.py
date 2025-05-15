import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graphs"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"
MEAN_SPEEDUP_FOLDER = "mean-speedup"

NTHREADS = []
METHODS = []

DEFAULT_METHOD = "taco_outer"
UNUSED_METHODS = ["taco_gustavson", "taco_inner"]
RESULT_FILE = "spgemm_results.json"

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
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    json_results = json.load(open(os.path.join(RESULTS_FOLDER, RESULT_FILE), "r"))
    num_threads = 1
    methods = set()
    for r in json_results:
        method = r["method"]
        if method in UNUSED_METHODS:
            continue
        else:
            methods.add(method)
        mtx = r["matrix"]
        time = r["time"]
        num_thread = r["num_threads"]
        results[mtx][method][num_thread] = time
        num_threads = max(num_threads, num_thread)

    global NTHREADS
    NTHREADS = [i + 1 for i in range(num_threads)]
    global METHODS
    METHODS = sorted(list(methods))
    return results


def plot_speedup_result(results, matrix, save_location):
    plt.figure(figsize=(10, 10))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [
                results[matrix][DEFAULT_METHOD][n_thread]
                / results[matrix][method][n_thread]
                for n_thread in NTHREADS
            ],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Speedup for {matrix} (with respect to {DEFAULT_METHOD})")
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)
    plt.close()


def plot_runtime_result(results, matrix, save_location):
    plt.figure(figsize=(10, 10))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [results[matrix][method][n_thread] for n_thread in NTHREADS],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Runtime for {matrix}")
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
        for matrix in results.keys():
            for i, n_thread in enumerate(NTHREADS):
                speedups[i] *= (
                    results[matrix][DEFAULT_METHOD][n_thread]
                    / results[matrix][method][n_thread]
                )

        mean_speedups = [speedup ** (1 / len(results)) for speedup in speedups]
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
        for matrix in results.keys():
            for i, n_thread in enumerate(NTHREADS):
                speedups[i] *= (
                    results[matrix][DEFAULT_METHOD][n_thread]
                    / results[matrix][method][n_thread]
                )

        mean_speedups = [speedup ** (1 / len(results)) for speedup in speedups]
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
    for matrix in results.keys():
        plot_speedup_result(
            results,
            matrix,
            os.path.join(
                GRAPH_FOLDER, SPEEDUP_FOLDER, f"{matrix.replace('/', '-')}.png"
            ),
        )
        plot_runtime_result(
            results,
            matrix,
            os.path.join(
                GRAPH_FOLDER, RUNTIME_FOLDER, f"{matrix.replace('/', '-')}.png"
            ),
        )

    plot_mean_speedup_result(results, os.path.join(GRAPH_FOLDER, "mean-speedup.png"))

    plot_mean_speedup_separate_result(
        results, os.path.join(GRAPH_FOLDER, MEAN_SPEEDUP_FOLDER)
    )

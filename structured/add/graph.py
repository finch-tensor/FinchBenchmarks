import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
RUNTIME_FOLDER = "runtime"
RESULTS_FOLDER = "results"

NTHREADS = [2**i for i in range(5)]

METHODS = [
    "shard_impl",
    # "cv_impl", 
]

DATASETS = {
    "image": [
        "www.abalip.com.jpg",
        "www.carmelmusic.com.jpg",
        "www.claudiozappi.it.jpg",
        "www.duo-thais.com.jpg", 
        "www.handball-riehen.ch.jpg",
        # "Fig0227(a)(washington_infrared).png",
        # "Fig1001(b)(edge_image).png",
        # "Fig1213(e)(Mask_B1_without_numbers).png",
        # "FigP0311.png",
    ],
}

COLORS = ["red", "gray", "cadetblue", "saddlebrown", "navy", "orange","black"]

def human_readable(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.0f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    else:
        return str(n)

def format_sparsity(x: float) -> str:
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return s


def sanitize_matrix_name(matrix: str) -> str:
    return (
        matrix.replace("/", "-")
        .replace(" + ", "_plus_")
        .replace(" ", "_")
    )


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/add_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            m = result["matrix"]

            if isinstance(m, str):
                matrix = m
            elif isinstance(m, dict):
                matrix = f"{m['size']}-{format_sparsity(m['sparsity'])}"
            else:
                raise TypeError(f"Unknown matrix format: {type(m)}")

            combine_results[result["dataset"]][matrix][result["method"]][
                result["n_threads"]
            ] = result["time"]

    return combine_results


# def plot_speedup_result(results, dataset, matrix, save_location):
#     plt.figure(figsize=(10, 6))
#     for method, color in zip(METHODS, COLORS):
#         plt.plot(
#             NTHREADS,
#             [
#                 results[dataset][matrix][DEFAULT_METHOD][n_thread]
#                 / results[dataset][matrix][method][n_thread]
#                 for n_thread in NTHREADS
#             ],
#             label=method,
#             color=color,
#             marker="o",
#             linestyle="-",
#             linewidth=1,
#         )

#     plt.title(
#         f"Structured Add - Speedup for {dataset}: {matrix} (with respect to {DEFAULT_METHOD})"
#     )
#     # plt.yscale("log", base=10)
#     plt.xticks(NTHREADS)
#     plt.xlabel("Number of Threads")
#     plt.ylabel(f"Speedup")

#     plt.legend()
#     plt.savefig(save_location)


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

    pretty_matrix = sanitize_matrix_name(matrix)

    plt.title(f"Structured Add - Runtime for {dataset}: {matrix}")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Runtime (in seconds)")

    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    plt.legend()
    plt.savefig(save_location)


# def weak_scaling_plot(results, dataset, save_location):
#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         NTHREADS,
#         [
#             results[dataset][f"{4096 * n_thread}-0.1"][SHARD_METHOD][n_thread]
#             for n_thread in NTHREADS
#         ],
#         label="shard_implementation",
#         color="grey",
#         marker="o",
#         linestyle="-",
#         linewidth=1,
#     )

#     plt.title(f"Structured Add - Weak Scaling with 10,000 x 4096 matrix per thread for {dataset}")
#     plt.xscale("log", base=2)
#     plt.xticks(NTHREADS)
#     plt.xlabel("Number of Threads")
#     plt.ylabel(f"Runtime (in seconds)")

#     plt.legend()
#     plt.savefig(save_location)
    


if __name__ == "__main__":
    os.makedirs(os.path.join(GRAPH_FOLDER, SPEEDUP_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(GRAPH_FOLDER, RUNTIME_FOLDER), exist_ok=True)

    results = load_json()
    for dataset, matrices in DATASETS.items():
        for matrix in matrices:
            plot_runtime_result(
                results,
                dataset,
                matrix,
                os.path.join(
                    GRAPH_FOLDER,
                    RUNTIME_FOLDER,
                    f"{dataset}-{sanitize_matrix_name(matrix)}.png",
                ),
            )

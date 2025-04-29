import json
import math
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE_PATH = "spadd_results.json"
CHARTS_DIRECTORY = "charts"


def get_num_threads_used():
    results = json.load(open(RESULTS_FILE_PATH, "r"))
    num_threads_used = set()
    for result in results:
        num_threads_used.add(result["num_threads"])
    return list(sorted(num_threads_used))


def all_formats_chart():
    methods = [
        "finch_static_schedule",
        "finch_greedy_schedule",
        "finch_julia_schedule",
    ]  # , "suite_sparse"]
    legend_labels = [
        "Finch Static Scheduler",
        "Finch Greedy Scheduler (chk_size = 100)",
        "Finch Julia Scheduler (chk_size = 100)",
    ]  # , "SuiteSparse"]

    colors = {
        "finch_static_schedule": "tab:green",
        "finch_greedy_schedule": "tab:gray",
        "finch_julia_schedule": "tab:blue",
    }

    results = json.load(open(RESULTS_FILE_PATH, "r"))
    for num_threads in get_num_threads_used():
        data = defaultdict(lambda: defaultdict(int))

        for result in results:
            if result["num_threads"] != num_threads:
                continue

            mtx = result["matrix"]
            method = result["method"]
            time = result["time"]
            data[mtx][method] = time

        for mtx, times in data.items():
            ref_time = times["taco"]
            for method, time in times.items():
                times[method] = ref_time / time

        ordered_data = sorted(
            data.items(),
            key=lambda mtx_results: (mtx_results[1]["taco"],),
            reverse=True,
        )

        all_data = defaultdict(list)
        for i, (mtx, times) in enumerate(ordered_data):
            for method in methods:
                all_data[method].append(
                    times.get(method, 0)
                )  # Use None or 0 as default

        ordered_mtxs = [mtx for mtx, _ in ordered_data]
        short_mtxs = [mtx.rsplit("/", 1)[-1] for mtx in ordered_mtxs]
        new_mtxs = {
            "uniform_dense": "uniform_dense",
            "uniform_sparse": "uniform_sparse",
        }
        short_mtxs = [new_mtxs.get(mtx, mtx) for mtx in short_mtxs]

        make_grouped_bar_chart(
            methods,
            short_mtxs,
            all_data,
            colors=colors,
            title=f"spadd Performance (Speedup Over Taco) with {num_threads} threads",
            legend_labels=legend_labels,
        )


def make_grouped_bar_chart(
    labels,
    x_axis,
    data,
    colors=None,
    labeled_groups=[],
    title="",
    y_label="",
    bar_labels_dict={},
    legend_labels=None,
    reference_label="",
):
    x = np.arange(len(data[labels[0]]))
    width = 0.22
    width = 0.8 / len(labels)
    multiplier = 0
    max_height = 0

    fig, ax = plt.subplots(figsize=(12, 4))
    for label in labels:
        label_data = data[label]
        max_height = max(max_height, max(label_data))
        offset = width * multiplier
        if colors:
            rects = ax.bar(
                x + offset, label_data, width, label=label, color=colors[label]
            )
        else:
            rects = ax.bar(x + offset, label_data, width, label=label)
        bar_labels = (
            bar_labels_dict[label]
            if (label in bar_labels_dict)
            else [
                round(float(val), 2) if label in labeled_groups else ""
                for val in label_data
            ]
        )
        ax.bar_label(rects, padding=0, labels=bar_labels, fontsize=5, rotation=90)
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1) / 2, x_axis)
    ax.tick_params(axis="x", which="major", labelsize=6, labelrotation=90)
    if legend_labels:
        ax.legend(legend_labels, loc="upper left", ncols=3, fontsize="small")
    else:
        ax.legend(loc="upper left", ncols=3, fontsize="small")
    ax.set_ylim(0, max_height + 0.5)

    # Adjusting x-axis limits to make bars go to the edges
    ax.set_xlim(-0.5, len(x_axis) - 0.5 + width * len(labels))

    plt.plot(
        [-1, len(x_axis)],
        [1, 1],
        linestyle="--",
        color="tab:red",
        linewidth=0.75,
        label=reference_label,
    )

    fig_file = title.lower().replace(" ", "_") + ".png"
    os.makedirs(CHARTS_DIRECTORY, exist_ok=True)
    plt.savefig(os.path.join(CHARTS_DIRECTORY, fig_file), dpi=200, bbox_inches="tight")
    plt.close()


all_formats_chart()

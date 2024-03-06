import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from collections import defaultdict

RESULTS_FILE_PATH = "spmv_results_lanka.json"
CHARTS_DIRECTORY = "spmv/charts_lanka2/"

def all_formats_chart():
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    mtxs = []
    data = defaultdict(list)
    finch_formats = get_best_finch_format()

    for result in results:
        mtx = result["matrix"]
        method = result["method"]
        if mtx not in mtxs:
            mtxs.append(mtx)

        if "finch" in method and finch_formats[mtx] != method:
            continue
        method = "finch" if "finch" in method else method
        data[method].append(result["time"])

    methods = ["julia_stdlib", "finch", "taco", "suite_sparse"]
    ref_data = data["taco"]
    for method in methods:
        method_data = data[method]
        data[method] = [ref_data[i] / method_data[i] for i in range(len(ref_data))] 

    make_grouped_bar_chart(methods, mtxs, data, labeled_groups=["finch"], title="SpMV Performance")


def get_best_finch_format():
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    formats = defaultdict(list)
    for result in results:
        if "finch" not in result["method"]:
            continue
        formats[result["matrix"]].append((result["method"], result["time"]))

    best_formats = defaultdict(list)
    for matrix, format_times in formats.items():
        best_format, _ = min(format_times, key=lambda x: x[1])
        best_formats[matrix] = best_format
    
    return best_formats


def get_method_results(method, mtxs=[]):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    mtx_times = {}
    for result in results:
        if result["method"] == method and (mtxs == [] or result["matrix"] in mtxs):
            mtx_times[result["matrix"]] = result["time"]
    return mtx_times


def get_speedups(faster_results, slower_results):
    speedups = {}
    for mtx, slow_time in slower_results.items():
        if mtx in faster_results:
            speedups[mtx] = slow_time / faster_results[mtx]
    return speedups


def order_speedups(speedups):
    ordered = [(mtx, time) for mtx, time in speedups.items()]
    return sorted(ordered, key=lambda x: x[1], reverse=True)

def method_to_ref_comparison_chart(method, ref, title=""):
    method_results = get_method_results(method)
    ref_results = get_method_results("taco")
    speedups = get_speedups(method_results, ref_results)

    x_axis = []
    data = defaultdict(list)
    for matrix, speedup in speedups.items():
        x_axis.append(matrix)
        data[method].append(speedup)
        data[ref].append(1)

    make_grouped_bar_chart([method, ref], x_axis, data, labeled_groups=[method], title=title)


def make_grouped_bar_chart(labels, x_axis, data, labeled_groups = [], title = "", y_label = ""):
    x = np.arange(len(data[labels[0]]))
    width = 0.2 
    multiplier = 0
    max_height = 0

    fig, ax = plt.subplots()
    for label, label_data in data.items():
        max_height = max(max_height, max(label_data))
        offset = width * multiplier
        rects = ax.bar(x + offset, label_data, width, label=label)
        bar_labels = [round(float(val), 2) if label in labeled_groups else "" for val in label_data]
        ax.bar_label(rects, padding=3, labels=bar_labels, fontsize=6)
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1)/2, x_axis)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, math.ceil(max_height) + 1)

    fig_file = title.lower().replace(" ", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200, bbox_inches="tight")


all_formats_chart()
method_to_ref_comparison_chart("finch", "taco", title="Finch SparseList Symmetric SpMV Performance")
method_to_ref_comparison_chart("finch_unsym", "taco", title="Finch SparseList SpMV Performance")
method_to_ref_comparison_chart("finch_vbl", "taco", title="Finch SparseVBL Symmetric SpMV Performance")
method_to_ref_comparison_chart("finch_vbl_unsym", "taco", title="Finch SparseVBL SpMV Performance")
method_to_ref_comparison_chart("finch_band", "taco", title="Finch SparseBand Symmetric SpMV Performance")
method_to_ref_comparison_chart("finch_band_unsym", "taco", title="Finch SparseBand SpMV Performance")
method_to_ref_comparison_chart("finch_pattern", "taco", title="Finch SparseList Pattern Symmetric SpMV Performance")
method_to_ref_comparison_chart("finch_pattern_unsym", "taco", title="Finch SparseList Pattern SpMV Performance")
method_to_ref_comparison_chart("finch_point", "taco", title="Finch SparsePoint SpMV Performance")



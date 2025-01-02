import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from collections import defaultdict
import re
RESULTS_FILE_PATH = "spmv_results.json"
CHARTS_DIRECTORY = "charts/"
FORMAT_ORDER = {
    "finch_sym_sparselist": -1,
    "finch_col_maj_sparselist": -2,
    "finch_row_maj_sparselist": -3,
    "finch_sym_sparseblocklist": -4,
    "finch_col_maj_sparseblocklist": -5,
    "finch_row_maj_sparseblocklist": -6,
    "finch_sym_sparseband": -7,
    "finch_col_maj_sparseband": -8,
    "finch_row_maj_sparseband": -9,
    "finch_sym_sparselist_pattern": -10,
    "finch_col_maj_sparselist_pattern": -11,
    "finch_row_maj_sparselist_pattern": -12,
    "finch_col_maj_sparsepoint_pattern": -13,
    "finch_row_maj_sparsepoint_pattern": -14,
}
FORMAT_LABELS = {
    "finch_sym_sparselist": "Symmetric SparseList",
    "finch_col_maj_sparselist": "SparseList",
    "finch_row_maj_sparselist": "SparseList (Row-Major)",
    "finch_sym_sparseblocklist": "Symmetric SparseVBL",
    "finch_col_maj_sparseblocklist": "SparseVBL",
    "finch_row_maj_sparseblocklist": "SparseVBL (Row-Major)",
    "finch_sym_sparseband": "Symmetric SparseBand",
    "finch_col_maj_sparseband": "SparseBand",
    "finch_row_maj_sparseband": "SparseBand (Row-Major)",
    "finch_sym_sparselist_pattern": "Symmetric Pattern",
    "finch_col_maj_sparselist_pattern": "Pattern",
    "finch_row_maj_sparselist_pattern": "Pattern (Row-Major)",
    "finch_col_maj_sparsepoint_pattern": "SparsePoint Pattern",
    "finch_row_maj_sparsepoint_pattern": "SparsePoint Pattern (Row-Major)",
}

def all_formats_chart(ordered_by_format=False):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = defaultdict(lambda: defaultdict(int))
    finch_formats = get_best_finch_format()

    for result in results:
        mtx = result["matrix"]
        method = result["method"]
        time = result["time"]

        if (method == "finch_col_maj_sparselist") or (method == "finch_row_maj_sparselist"):
            # finch_baseline_time = data[mtx]["finch_baseline"]
            # data[mtx]["finch_baseline"] = time if finch_baseline_time == 0 else min(time, finch_baseline_time)
            #infinity in python is float('inf')
            #data[mtx]["finch_baseline"] = min(data[mtx].get("finch_baseline", float('inf')), data[mtx][method])
            #the above line doesn't work, but this one does:
            data[mtx]["finch_baseline"] = min(data[mtx].get("finch_baseline", float('inf')), time)
        if "finch" in method and finch_formats[mtx] != method:
            continue
        method = "finch" if "finch" in method else method
        data[mtx][method] = time

    for mtx, times in data.items():
        ref_time = times["taco_row_maj"]
        for method, time in times.items():
            times[method] = ref_time / time

    if ordered_by_format:
        #ordered_data = sorted(data.items(), key = lambda mtx_results: (mtx_results[1]["finch"] > 1, FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
        ordered_data = sorted(data.items(), key = lambda mtx_results: (FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
    else:
        ordered_data = sorted(data.items(), key = lambda mtx_results: mtx_results[1]["finch"], reverse=True)

    methods = ["finch", "finch_baseline", "julia_stdlib", "mkl", "eigen", "cora"]#, "suite_sparse"]
    legend_labels = ["Finch (Best)", "Finch (Baseline)", "Julia Stdlib", "MKL", "Eigen", "CoRa"]#, "SuiteSparse"]

    colors = {
        "finch": "tab:green",
        "finch_baseline": "tab:gray",
        "julia_stdlib": "tab:blue",
        "mkl": "tab:orange",
        "eigen": "tab:red",
        "cora": "tab:purple"
    }

    all_data = defaultdict(list)
    for i, (mtx, times) in enumerate(ordered_data):
        for method in methods:
            all_data[method].append(times.get(method, 0))  # Use None or 0 as default

    ordered_mtxs = [mtx for mtx, _ in ordered_data]
    labels = [FORMAT_LABELS[finch_formats[mtx]] for mtx, _ in ordered_data]
    short_mtxs = [mtx.rsplit('/',1)[-1] for mtx in ordered_mtxs]
    new_mtxs = {
        "toeplitz_large_sparseband_sym": "large_sparseband_sym",
        "toeplitz_medium_sparseband_sym": "medium_sparseband_sym",
        "toeplitz_small_sparseband_sym": "small_sparseband_sym",
        #"TSOPF_RS_b678_c1": "*RS_b678_c1",
    }
    short_mtxs = [new_mtxs.get(mtx, mtx) for mtx in short_mtxs]

    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, labeled_groups=["finch"], bar_labels_dict={"finch": labels[:]}, title="SpMV Performance (Speedup Over Taco) labeled", legend_labels=legend_labels)
    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, title="SpMV Performance (Speedup Over Taco)", legend_labels=legend_labels)

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


def all_formats_for_matrix_chart(matrix):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = {}
    for result in results:
        if result["matrix"] == matrix:
            data[result["method"]] = result["time"]
    
    formats = []
    speedups = []
    bar_colors = []
    for format, time in data.items():
        formats.append(format)
        speedups.append(data["taco"] / time)
        bar_colors.append("orange" if "finch" in format else "green")
    
    fig, ax = plt.subplots()
    ax.bar(formats, speedups, width = 0.2, color = bar_colors)
    ax.set_ylabel("Speedup")
    ax.set_title(matrix + " Performance")
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)

    fig_file = matrix.lower().replace("/", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + "/matrices/" + fig_file, dpi=200, bbox_inches="tight")
    plt.close() 


def make_grouped_bar_chart(labels, x_axis, data, colors = None, labeled_groups = [], title = "", y_label = "", bar_labels_dict={}, legend_labels=None, reference_label = ""):
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
            rects = ax.bar(x + offset, label_data, width, label=label, color=colors[label])
        else:
            rects = ax.bar(x + offset, label_data, width, label=label)
        bar_labels = bar_labels_dict[label] if (label in bar_labels_dict) else [round(float(val), 2) if label in labeled_groups else "" for val in label_data]
        ax.bar_label(rects, padding=0, labels=bar_labels, fontsize=5, rotation=90)
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1)/2, x_axis)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)
    if legend_labels:
        ax.legend(legend_labels, loc='upper left', ncols=3, fontsize='small')
    else:
        ax.legend(loc='upper left', ncols=3, fontsize='small')
    ax.set_ylim(0, max_height + 0.5)

        # Adjusting x-axis limits to make bars go to the edges
    ax.set_xlim(-0.5, len(x_axis) - 0.5 + width * len(labels))

    plt.plot([-1, len(x_axis)], [1, 1], linestyle='--', color="tab:red", linewidth=0.75, label=reference_label)

    fig_file = title.lower().replace(" ", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200, bbox_inches="tight")
    plt.close()
    

all_formats_chart()
all_formats_chart(ordered_by_format=True)
# method_to_ref_comparison_chart("finch", "taco", title="Finch SparseList Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_col_maj_sparselist", "taco", title="Finch SparseList SpMV Performance")
# method_to_ref_comparison_chart("finch_row_maj_sparselist", "taco", title="Finch SparseList Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseblocklist_sym", "taco", title="Finch SparseVBL Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseblocklist_col_maj_sparselist", "taco", title="Finch SparseVBL SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseblocklist_row_maj_sparselist", "taco", title="Finch SparseVBL Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseband_sym", "taco", title="Finch SparseBand Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseband_col_maj_sparselist", "taco", title="Finch SparseBand SpMV Performance")
# method_to_ref_comparison_chart("finch_sparseband_row_maj_sparselist", "taco", title="Finch SparseBand Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_sparselist_pattern_sym", "taco", title="Finch SparseList Pattern Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_sparselist_pattern_col_maj_sparselist", "taco", title="Finch SparseList Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_sparselist_pattern_row_maj_sparselist", "taco", title="Finch SparseList Pattern Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_point", "taco", title="Finch SparsePoint SpMV Performance")
# method_to_ref_comparison_chart("finch_point_row_maj_sparselist", "taco", title="Finch SparsePoint Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_sparsepoint_pattern_col_maj_sparselist", "taco", title="Finch SparsePoint Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_sparsepoint_pattern_row_maj_sparselist", "taco", title="Finch SparsePoint Pattern Row Major SpMV Performance")
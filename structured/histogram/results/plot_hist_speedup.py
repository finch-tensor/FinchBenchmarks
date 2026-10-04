import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})

def matrix_name(m):
    name = m.split("/")[-1].rsplit(".", 1)[0]
    return name[4:] if name.startswith("www.") else name

FILES = {
    "halide_rfactor_hist": f"./halide_rfactor_hist_{THREADS}_threads.json",
    "halide_hist":   f"./halide_hist_{THREADS}_threads.json",
    "wingspan_hist": f"./wing_hist_{THREADS}_threads.json",
    "birdseed_hist": f"./birdseed_hist_{THREADS}_threads.json",
}

# image -> method -> time (s)
data: dict = {}
for method, path in FILES.items():
    try:
        with open(path) as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"warning: {path} not found, plotting without {method}")
        continue
    for entry in entries:
        data.setdefault(matrix_name(entry["matrix"]), {})[entry["method"]] = entry["time"]

# halide-rfactor is the baseline at 1.0; drop series with no results at all
BASELINE = "halide_rfactor_hist"
images  = sorted(mat for mat in data if BASELINE in data[mat])
methods = [meth for meth in FILES if any(meth in data[mat] for mat in images)]

speedups: dict = {m: [] for m in methods}
for mat in images:
    base_t = data[mat][BASELINE]
    for meth in methods:
        t = data[mat].get(meth)
        speedups[meth].append(base_t / t if t else float("nan"))

LABELS = {
    "halide_rfactor_hist": "halide-rfactor",
    "halide_hist":   "halide-atomics",
    "wingspan_hist": "wingspan",
    "birdseed_hist": "birdseed",
}
# Wong (2011) colorblind-safe palette
COLORS = {
    "halide_rfactor_hist": "#56B4E9",
    "halide_hist":   "#CC79A7",
    "wingspan_hist": "#E69F00",
    "birdseed_hist": "#009E73",
}

n_images  = len(images)
n_methods = len(methods)
bar_width = 0.8 / n_methods
group_gap = 0.06
x = np.arange(n_images)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, meth in enumerate(methods):
    offsets = x + (i - n_methods / 2 + 0.5) * (bar_width + group_gap / n_methods)
    ax.bar(
        offsets,
        speedups[meth],
        width=bar_width,
        label=LABELS[meth],
        color=COLORS[meth],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(images, rotation=15, ha="right")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_locator(ticker.NullLocator())
# leave headroom above the tallest bar for the legend
ax.set_ylim(top=np.nanmax([s for v in speedups.values() for s in v]) * 1.6)
ax.set_ylabel(f"Speedup over {LABELS[BASELINE]}")
ax.set_title(f"Structured Histogram Speedup ({THREADS} threads)", fontweight="bold", pad=14)
ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper center", ncol=n_methods, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./hist_speedup.png", dpi=200)
print("Saved.")

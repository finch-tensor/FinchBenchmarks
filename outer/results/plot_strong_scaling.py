import json
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})

# MKL runs in both sweeps as an accuracy reference; plot the birdseed run's MKL.
SOURCES = {
    "./outer_wingspan_threads_*.json": {"wingspan_outer": "wingspan"},
    "./outer_birdseed_threads_*.json": {"birdseed_outer": "birdseed", "mkl": "mkl"},
}

# Build dict: method -> {threads -> time_ms}
data: dict = {}
for pattern, rename in SOURCES.items():
    for path in glob.glob(pattern):
        with open(path) as f:
            for entry in json.load(f):
                method = entry["method"]
                if method not in rename:
                    continue
                t_ms = entry["time"] * 1000
                threads = entry["n_threads"]
                data.setdefault(rename[method], {})[threads] = t_ms

methods = [m for m in ["wingspan", "birdseed", "mkl"] if m in data]
all_threads = sorted({t for m in data.values() for t in m})

# Wong (2011) colorblind-safe palette
COLORS = {
    "wingspan": "#E69F00",
    "birdseed": "#009E73",
    "mkl":      "#56B4E9",
}
MARKERS = {
    "wingspan": "o",
    "birdseed": "^",
    "mkl":      "s",
}


fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for meth in methods:
    xs = sorted(data[meth].keys())
    ys = [data[meth][x] for x in xs]
    ax.plot(
        xs, ys,
        label=meth,
        color=COLORS[meth],
        marker=MARKERS[meth],
        linewidth=1.8,
        markersize=6,
        zorder=3,
    )

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(all_threads)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# Tighten y-axis around the data range
all_times = [t for m in data.values() for t in m.values()]
ax.set_ylim(min(all_times) * 0.8, max(all_times) * 1.2)

ax.set_xlabel("Number of Threads")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Outer Product Strong Scaling", fontweight="bold", pad=14)
ax.grid(axis="both", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="both", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper right", framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./strong_scaling.png", dpi=200)
print("Saved.")

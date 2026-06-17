import json
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})

# ── Data ────────────────────────────────────────────────────────────────────

def matrix_name(m):
    name = m.split("/")[-1].rsplit(".", 1)[0]
    return name[4:] if name.startswith("www.") else name

# Load all thread files
records = []
for path in glob.glob("./add_*_threads.json"):
    with open(path) as f:
        records.extend(json.load(f))

# Build dict: matrix -> {threads -> time_ms}  (shard_impl only)
data: dict = {}
for entry in records:
    if entry["method"] == "cv_impl":
        continue
    name = matrix_name(entry["matrix"])
    t_ms = entry["time"] * 1000
    threads = entry["n_threads"]
    data.setdefault(name, {})[threads] = t_ms

matrices    = sorted(data.keys())
all_threads = sorted({t for m in data.values() for t in m})

# Wong (2011) colorblind-safe palette
COLORS = {
    "abalip.com":         "#E69F00",
    "carmelmusic.com":    "#56B4E9",
    "claudiozappi.it":    "#009E73",
    "duo-thais.com":      "#CC79A7",
    "handball-riehen.ch": "#0072B2",
}
MARKERS = {
    "abalip.com":         "o",
    "carmelmusic.com":    "s",
    "claudiozappi.it":    "^",
    "duo-thais.com":      "D",
    "handball-riehen.ch": "P",
}

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for mat in matrices:
    xs = sorted(data[mat].keys())
    ys = [data[mat][x] for x in xs]
    ax.plot(
        xs, ys,
        label=mat,
        color=COLORS.get(mat, "#333333"),
        marker=MARKERS.get(mat, "o"),
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
ax.set_title("Structured Add Strong Scaling", fontweight="bold", pad=14)
ax.grid(axis="both", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="both", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), framealpha=0.85, edgecolor="#cccccc", ncol=2)

plt.tight_layout()
plt.savefig("./add_strong_scaling.png", dpi=200)
print("Saved.")

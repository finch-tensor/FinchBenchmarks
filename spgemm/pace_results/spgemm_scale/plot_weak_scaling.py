import json
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ────────────────────────────────────────────────────────────────────

RENAME = {
    "spgemm_finch_gustavson_dynamic": "wingspan_gustavson",
    "spgemm_eigen":                   "eigen",
    "spgemm_mkl":                     "mkl",
}

records = []
for path in glob.glob("./spgemm_scale_threads_weak_*.json"):
    with open(path) as f:
        records.extend(json.load(f))

data: dict = {}
for entry in records:
    method = entry["method"]
    if method not in RENAME:
        continue
    name = RENAME[method]
    t_ms = entry["time"] * 1000
    threads = entry["threads"]
    data.setdefault(name, {})[threads] = t_ms

methods     = ["wingspan_gustavson", "mkl", "eigen"]
all_threads = sorted({t for m in data.values() for t in m})

COLORS  = {
    "wingspan_gustavson": "#E69F00",
    "mkl":                "#56B4E9",
    "eigen":              "#009E73",
}
MARKERS = {
    "wingspan_gustavson": "o",
    "mkl":                "s",
    "eigen":              "^",
}

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
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

all_times = [t for m in data.values() for t in m.values()]
ax.set_ylim(min(all_times) * 0.8, max(all_times) * 1.2)

ax.set_xlabel("Number of Threads", fontsize=11)
ax.set_ylabel("Runtime (ms)", fontsize=11)
ax.set_title("SpGEMM Weak Scaling", fontsize=13, fontweight="bold", pad=14)
ax.grid(axis="both", which="major", color="#cccccc", linewidth=0.7, zorder=0)
ax.grid(axis="both", which="minor", color="#e8e8e8", linewidth=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("./weak_scaling.png", dpi=150, bbox_inches="tight")
print("Saved.")

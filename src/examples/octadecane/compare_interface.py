import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)

paths = sorted(
    glob.glob("./data/checkpoint_*.npz"),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry

scale = 1.0 / 884.0


def load_and_sort(path, open_curve=False):
    data = np.load(path)
    x = data["x"] * scale
    y = data["y"] * scale
    pts = np.column_stack((x, y))
    n = len(pts)
    if n <= 2:
        return x, y
    ordered = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    ordered[0] = 0
    visited[0] = True
    for i in range(1, n):
        dists = np.sum((pts[ordered[i - 1]] - pts[~visited]) ** 2, axis=1)
        next_idx = np.where(~visited)[0][np.argmin(dists)]
        ordered[i] = next_idx
        visited[next_idx] = True
    x_sorted = x[ordered]
    y_sorted = y[ordered]

    if open_curve and n > 2:
        dists = np.sqrt(np.diff(x_sorted) ** 2 + np.diff(y_sorted) ** 2)
        median_dist = np.median(dists)
        max_gap_idx = np.argmax(dists)
        if dists[max_gap_idx] > 2.0 * median_dist:
            x_sorted = np.insert(x_sorted, max_gap_idx + 1, np.nan)
            y_sorted = np.insert(y_sorted, max_gap_idx + 1, np.nan)

    return x_sorted, y_sorted


x_danaila_800, y_danaila_800 = load_and_sort(
    "./data/other_authors/danaila_800.npz", open_curve=True
)
x_okada_800, y_okada_800 = load_and_sort("./data/other_authors/okada_800.npz")
x_danaila_1575, y_danaila_1575 = load_and_sort("./data/other_authors/danaila_1575.npz")
x_okada_1575, y_okada_1575 = load_and_sort("./data/other_authors/okada_1575.npz")
x_wang_1575, y_wang_1575 = load_and_sort("./data/other_authors/wang_1575.npz")

fig, ax = plt.subplots(figsize=(5.0, 5.0))

boundaries = []
for file_path in paths:
    data = np.load(file_path)
    u = data["u"]
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    x_b = np.asarray(X_b) / cfg.l
    y_b = np.asarray(Y_b) / cfg.l
    ax.plot(x_b, y_b, linestyle="--", color="C0")
    boundaries.append((x_b, y_b))

ax.plot(x_danaila_800, y_danaila_800, linestyle="-", color="C1")
ax.plot(x_danaila_1575, y_danaila_1575, linestyle="-", color="C1")
ax.plot(x_okada_800, y_okada_800, linestyle="-", color="C2")
ax.plot(x_okada_1575, y_okada_1575, linestyle="-", color="C2")
ax.plot(x_wang_1575, y_wang_1575, linestyle="-", color="C3")

for i, label in zip([0, -1], [r"$\tilde{t} = 0.032$", r"$\tilde{t} = 0.063$"]):
    x_b, y_b = boundaries[i]
    ax.text(x_b.max() - 0.05, y_b[np.argmax(x_b)] - 0.2, label, va="center", rotation=60)

legend_elements = [
    mlines.Line2D([], [], linestyle="--", color="C0", label="Present work"),
    mlines.Line2D(
        [],
        [],
        linestyle="-",
        color="C1",
        label="Rakotondrandisa et al. (2019)",
    ),
    mlines.Line2D([], [], linestyle="-", color="C2", label="Okada (1984)"),
    mlines.Line2D([], [], linestyle="-", color="C3", label="Wang et al. (2010)"),
]

ax.legend(handles=legend_elements, loc="best", fontsize=11)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("./graphs/compared.png")

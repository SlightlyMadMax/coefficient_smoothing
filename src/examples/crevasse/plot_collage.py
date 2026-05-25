import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from src.parameters.config import ExperimentConfig
from src.core.constants import ABS_ZERO

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 0.8,
        "figure.dpi": 300,
    }
)

# ── Load Configs ──────────────────────────────────────────────────────────────
cfg_cond = ExperimentConfig.load_from_file("./conduction/config.json")
cfg_conv = ExperimentConfig.load_from_file("./convection/config.json")

CASES = {
    "Conduction": {
        "cfg": cfg_cond,
        "folder": "./data/conduction",
        "checkpoints": [600, 14400, 36000, 86400],
    },
    "Convection": {
        "cfg": cfg_conv,
        "folder": "./data/convection/48_hrs",
        "checkpoints": [12000, 288000, 720000, 1728000],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_u_celsius(folder: str, checkpoint: int, cfg: ExperimentConfig) -> np.ndarray:
    path = Path(folder) / f"checkpoint_{checkpoint}.npz"
    u = np.load(path)["u"]
    u_dim = u * cfg.delta_u + cfg.u_ref
    return u_dim + ABS_ZERO


def find_clim(*arrays: np.ndarray):
    return min(a.min() for a in arrays), max(a.max() for a in arrays)


# ── Load data ─────────────────────────────────────────────────────────────────
data = {}
for label, cfg_case in CASES.items():
    data[label] = [
        load_u_celsius(cfg_case["folder"], ck, cfg_case["cfg"])
        for ck in cfg_case["checkpoints"]
    ]

all_arrays = [arr for arrays in data.values() for arr in arrays]
vmin, vmax = find_clim(*all_arrays)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(6.3, 3.1),
    constrained_layout=True,
)
cmap = "Blues"
axis_ticks = [0.0, 0.05, 0.10, 0.15, 0.20]
row_labels = ["a", "b"]

for row_idx, (label, cfg_case) in enumerate(CASES.items()):
    cfg = cfg_case["cfg"]
    X, Y = cfg.geometry.mesh_grid
    arrays = data[label]

    for col_idx, (arr, step) in enumerate(zip(arrays, cfg_case["checkpoints"])):
        ax = axes[row_idx, col_idx]

        im = ax.pcolormesh(
            X,
            Y,
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        # Изотерма T = 0 °C
        ax.contour(X, Y, arr, levels=[0.0], colors="black", linewidths=0.8)

        ax.set_box_aspect(1)

        axis_ticks = [0.0, 0.1, 0.2]
        ax.xaxis.set_major_locator(ticker.FixedLocator(axis_ticks))
        ax.yaxis.set_major_locator(ticker.FixedLocator(axis_ticks))

        # Только левый столбец: подписи Y
        if col_idx == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

        # Только нижний ряд: подписи X
        if row_idx == 1:
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)

        # ── Формирование подписи времени ──
        t_sec = cfg.geometry.dt * step
        if t_sec >= 3600:
            time_val = int(round(t_sec / 3600))
            time_str = rf"$t = {time_val}$ h"
        elif t_sec >= 60:
            time_val = int(round(t_sec / 60))
            time_str = rf"$t = {time_val}$ min"
        else:
            time_val = int(round(t_sec))
            time_str = rf"$t = {time_val}$ s"

        ax.text(
            0.98,
            0.025,
            time_str,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="right",
        )

    left_ax = axes[row_idx, 0]
    left_ax.text(
        -0.3,
        0.5,
        row_labels[row_idx],
        transform=left_ax.transAxes,
        fontweight="bold",
        va="center",
        ha="right",
        clip_on=False,
    )

# Цветовая шкала
cbar_ticks = np.linspace(vmin, vmax, 6)
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02,
    label="Temperature, °C",
    ticks=cbar_ticks,
)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

fig.savefig("./graphs/collage.png", bbox_inches="tight", dpi=300)
plt.show()

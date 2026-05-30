import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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

ABS_ZERO = -273.15


cfg = ExperimentConfig.load_from_file("./config.json")

geometry = cfg.geometry

melting_steps = [75000, 432000, 800400]
freezing_steps = [4800, 108000, 800400]

fig, axes = plt.subplots(3, 2, figsize=(6.3, 3.15))
fig.subplots_adjust(hspace=0.15, wspace=0.2, right=0.88)

vmin, vmax = -5, 5
contours = []

for i, step in enumerate(melting_steps):
    ax = axes[i, 0]

    data_melting = np.load(f"./data/melting/checkpoint_{step}.npz")
    u_melting = data_melting["u"]

    u_dim = u_melting * cfg.delta_u + cfg.u_ref

    n_x, n_y = u_melting.shape[1], u_melting.shape[0]
    x = np.linspace(0, geometry.width * 100, n_x)
    y = np.linspace(0, geometry.height * 100, n_y)
    X, Y = np.meshgrid(x, y)

    temp_celsius = u_dim + ABS_ZERO

    contour = ax.contourf(
        X, Y, temp_celsius, levels=101, cmap="Blues", vmin=vmin, vmax=vmax
    )
    contours.append(contour)

    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
    X_b = [x * 100 for x in X_b]
    Y_b = [y * 100 for y in Y_b]
    ax.scatter(X_b, Y_b, s=0.1, color="black")

    if i == 2:
        ax.set_xlabel(r"$x$, cm")
    if i < 2:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$y$, cm")
    ax.set_xlim(0, geometry.width * 100)
    ax.set_ylim(0, geometry.height * 100)
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))

for i, step in enumerate(freezing_steps):
    ax = axes[i, 1]

    data_freezing = np.load(f"./data/freezing/checkpoint_{step}.npz")
    u_freezing = data_freezing["u"]

    u_dim = u_freezing * cfg.delta_u + cfg.u_ref

    n_x, n_y = u_freezing.shape[1], u_freezing.shape[0]
    x = np.linspace(0, geometry.width * 100, n_x)
    y = np.linspace(0, geometry.height * 100, n_y)
    X, Y = np.meshgrid(x, y)

    temp_celsius = u_dim + ABS_ZERO

    contour = ax.contourf(
        X, Y, temp_celsius, levels=101, cmap="Blues", vmin=vmin, vmax=vmax
    )

    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
    X_b = [x * 100 for x in X_b]
    Y_b = [y * 100 for y in Y_b]
    ax.scatter(X_b, Y_b, s=0.1, color="black")

    if i == 2:
        ax.set_xlabel(r"$x$, cm")
    if i < 2:
        ax.tick_params(labelbottom=False)
    ax.set_xlim(0, geometry.width * 100)
    ax.set_ylim(0, geometry.height * 100)
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))

# === Общие подписи столбцов ===
# Используем fig.text() с координатами относительно всей фигуры (0–1)
# x=0.27 — центр левого столбца, x=0.73 — центр правого
# y=1.02 — чуть выше верхней границы области графиков
fig.text(0.27, 0.9, "a", ha="center", va="bottom", fontsize=12, fontweight="bold")
fig.text(0.73, 0.9, "b", ha="center", va="bottom", fontsize=12, fontweight="bold")
# =============================

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contours[-1], cax=cbar_ax)
cbar.set_ticks(np.linspace(-5, 5, 11))
cbar.set_label("Temperature, °C")

plt.savefig("./graphs/boundary_evolution.tiff", dpi=300, bbox_inches="tight")
plt.show()
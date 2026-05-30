import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.core.constants import ABS_ZERO
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

# --------------------------------------------------
# Загрузка конфигураций
# --------------------------------------------------

cfg_8c: ExperimentConfig = ExperimentConfig.load_from_file("parameters/8c.json")
cfg_4c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/4c.json")
cfg_5pt6c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/5pt6c.json")


# --------------------------------------------------
# Загрузка данных и перевод температуры в размерный вид
# --------------------------------------------------

data_8c = np.load("./data/8c/checkpoint_750.npz")
u_8c_dim = data_8c["u"] * cfg_8c.delta_u + cfg_8c.u_ref + ABS_ZERO

data_4c = np.load("./data/4c/checkpoint_1800.npz")
u_4c_dim = data_4c["u"] * cfg_4c.delta_u + cfg_4c.u_ref + ABS_ZERO

data_5pt6c = np.load("./data/5pt6c/5pt6c_thin/checkpoint_9000.npz")
u_5pt6c_dim = data_5pt6c["u"] * cfg_5pt6c.delta_u + cfg_5pt6c.u_ref + ABS_ZERO


# --------------------------------------------------
# Универсальная функция подготовки поля
# --------------------------------------------------


def prepare_field(u_dim, geometry, x_min, x_max):
    ny, nx = u_dim.shape
    x = np.linspace(0.0, geometry.width, nx)
    y = np.linspace(0.0, geometry.height, ny)
    mask = (x >= x_min) & (x <= x_max)
    u_crop = u_dim[:, mask]
    X, Y = np.meshgrid(x[mask] - x_min, y)
    return X, Y, u_crop


# --------------------------------------------------
# Подготовка данных для трёх режимов
# --------------------------------------------------

X4, Y4, u4 = prepare_field(u_4c_dim, cfg_4c.geometry, 0.3, 0.5)
X56, Y56, u56 = prepare_field(u_5pt6c_dim, cfg_5pt6c.geometry, 0.3, 0.5)
X8, Y8, u8 = prepare_field(u_8c_dim, cfg_8c.geometry, 0.3, 0.5)


# --------------------------------------------------
# Общая цветовая шкала
# --------------------------------------------------

u_min = min(u4.min(), u56.min(), u8.min())
u_max = max(u4.max(), u56.max(), u8.max())

levels = np.linspace(u_min, u_max, 60)
ticks = np.linspace(u_min, u_max, 5)


# --------------------------------------------------
# Построение объединённого рисунка
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), constrained_layout=True)

datasets = [
    (X4, Y4, u4, "a", cfg_4c),
    (X56, Y56, u56, "b", cfg_5pt6c),
    (X8, Y8, u8, "c", cfg_8c),
]

for i, (ax, (X, Y, u, title, cfg)) in enumerate(zip(axes, datasets)):
    cf = ax.contourf(X, Y, u, levels=levels, cmap="Blues")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(0.0, cfg.geometry.height)

    ax.set_xticks([0.0, 0.1, 0.2])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(r"$x$, m")

    if i == 0:
        ax.set_yticks(np.linspace(0.0, cfg.geometry.height, 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel(r"$y$, m")
    else:
        ax.tick_params(labelleft=False)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.tick_params(labelsize=10)
    ax.contour(X, Y, u, levels=[0.05], colors="black", linewidths=0.8)


# --------------------------------------------------
# Общий colorbar
# --------------------------------------------------

cbar = fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label(
    r"Temperature, $^\circ\mathrm{C}$", rotation=270, labelpad=15
)
cbar.set_ticks(ticks)
cbar.ax.tick_params(labelsize=10)


# --------------------------------------------------
# Сохранение
# --------------------------------------------------

plt.savefig("./graphs/melting_regimes_v3.tiff", dpi=300)
plt.show()

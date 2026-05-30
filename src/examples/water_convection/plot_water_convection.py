import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.parameters.config import ExperimentConfig
from src.examples.water_convection.benchmark_solution import calculate_T_profile_Y05

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


# -----------------------------
# helper for subfigure labels
# -----------------------------
def add_subfigure_label(ax, label):
    """Add subfigure label (a), (b), etc. centered above the axes."""
    ax.text(
        0.5,
        1.05,
        f"{label}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=10,
    )


# -----------------------------
# load data
# -----------------------------
cfg = ExperimentConfig.load_from_file("./config.json")

data = np.load("./data/1st_order_bc/151x151/checkpoint_7200.npz")
u = data["u"]
v_x, v_y = data["v_x"], data["v_y"]

n_x, n_y = u.shape[1], u.shape[0]
x = np.linspace(0, 1, n_x)
y = np.linspace(0, 1, n_y)
X, Y = np.meshgrid(x, y)

# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 3.15), constrained_layout=True)

# -------- (а) field ----------
stride = 8
levels = np.linspace(0.0, 1.0, 101)
contour = ax0.contourf(X, Y, u, levels=levels, cmap="Blues")

ax0.quiver(
    X[::stride, ::stride],
    Y[::stride, ::stride],
    v_x[::stride, ::stride],
    v_y[::stride, ::stride],
    color="black",
    scale_units="xy",
)

ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.set_xlabel(r"$X$")
ax0.set_ylabel(r"$Y$")
ax0.set_aspect("equal", adjustable="box")

cbar = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.set_label("Dimensionless temperature", rotation=270, labelpad=15)

add_subfigure_label(ax0, "a")

# -------- (б) profile --------
u_true = calculate_T_profile_Y05(x)
u_mid = u[n_y // 2, :]
MARKER_STEP = 5

ax1.plot(x, u_true, color="#0072BD")
ax1.plot(
    x[::MARKER_STEP],
    u_mid[::MARKER_STEP],
    marker="o",
    markersize=4,
    markerfacecolor="#D95319",
    markeredgecolor="white",
    markeredgewidth=1.5,
    linestyle="",
)

L = 0.08
i1 = int(0.05 * (n_x - 1))
x1, y1 = x[i1], u_true[i1] - 0.02
ax1.plot([x1 + 0.01, x1 + 0.01 + L], [y1, y1], linewidth=0.8, color="black")
ax1.text(x1 + 0.01 + L + 0.01, y1, "1", va="center", fontsize=10)

target_x = 0.5
base_idx = int(target_x * (n_x - 1))
marker_idx = (base_idx // MARKER_STEP) * MARKER_STEP

x2, y2 = x[marker_idx], u_mid[marker_idx]
ax1.plot([x2, x2], [y2 + 0.01, y2 + 0.01 + L], linewidth=0.8, color="black")
ax1.text(x2, y2 + 0.01 + L + 0.01, "2", ha="center", fontsize=10)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"$X$")
ax1.set_ylabel("Θ")
ax1.set_aspect("equal", adjustable="box")

add_subfigure_label(ax1, "b")

# -----------------------------
plt.savefig("./graphs/compared.tiff", dpi=300)
# plt.show()

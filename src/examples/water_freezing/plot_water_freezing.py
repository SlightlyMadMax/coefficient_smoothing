import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.init_values import initialize_velocity
from src.fluid_dynamics.utils import calculate_velocity_from_sf
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
cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry
img = plt.imread("./data/kowalewski.png")
data = np.load("data/old_data/after_freezing_151x151.npz")
u = data["u"]
sf = data["sf"]
w = data["w"]
u_dim = u * cfg.delta_u + cfg.u_ref
v_x, v_y = initialize_velocity(geometry=geometry)
calculate_velocity_from_sf(sf, v_x, v_y, cfg)

n_x, n_y = u.shape[1], u.shape[0]
x = np.linspace(0, geometry.width, n_x)
y = np.linspace(0, geometry.height, n_y)
X, Y = np.meshgrid(x, y)

# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 3.15), constrained_layout=True)

# -------- (а) field ----------
ax0.imshow(img, extent=[0, geometry.width, 0, geometry.height])
X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
ax0.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2, label="Численное решение")

ax0.set_xlabel(r"$x$, m")
ax0.set_ylabel(r"$y$, m")
ax0.set_aspect("equal", adjustable="box")

add_subfigure_label(ax0, "a")

# -------- (б) profile --------
stride = 8
contour = ax1.contourf(X, Y, u_dim + ABS_ZERO, levels=101, cmap="Blues")

ax1.quiver(
    X[::stride, ::stride],
    Y[::stride, ::stride],
    v_x[::stride, ::stride],
    v_y[::stride, ::stride],
    color="black",
    scale_units="xy",
)

ax1.plot(X_b, Y_b, linestyle="--", color="k", linewidth=1.5)

ax1.set_xlabel(r"$x$, m")
ax1.set_ylabel(r"$y$, m")
ax1.set_aspect("equal", adjustable="box")

L = 0.003
dx_label = 0.0008
dy_label = -0.0005

# i2 = int(0.71 * (len(X_b) - 1))
# x2, y2 = X_b[i2], Y_b[i2]
# ax0.plot([x2, x2 + L], [y2, y2], color="black")
# ax0.text(x2 + L + dx_label, y2 + dy_label, "1", ha="center")

cbar = fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(-10, 10, 9))
cbar.set_label(r"Temperature, $^{\circ}\mathrm{C}$", rotation=270, labelpad=15)

add_subfigure_label(ax1, "b")

# -----------------------------
plt.savefig("./graphs/water_freezing_upd.tiff", dpi=300)
plt.show()

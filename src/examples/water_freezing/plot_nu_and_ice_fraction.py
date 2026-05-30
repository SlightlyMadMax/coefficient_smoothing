import matplotlib as mpl
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from src.parameters.config import ExperimentConfig
from src.utils.nusselt import calculate_nusselt


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


cfg = ExperimentConfig.load_from_file("./config.json")


def load_nusselt_data(data_mask: str, max_checkpoints: int = None):
    paths = sorted(
        glob.glob(data_mask),
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
    )

    times, nu_values = [], []
    count = 0
    for fp in paths:
        match = re.search(r"checkpoint_(\d+)\.npz", fp)
        if not match:
            continue
        n = int(match.group(1))
        if max_checkpoints is not None and count >= max_checkpoints:
            break
        u = np.load(fp)["u"]
        nu_values.append(calculate_nusselt(u=u, cfg=cfg, wall="left"))
        times.append(n * cfg.geometry.dt)
        count += 1
    return np.array(times), np.array(nu_values)


def load_ice_fraction_data(data_mask: str, max_checkpoints: int = None):
    paths = sorted(
        glob.glob(data_mask),
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
    )

    times, ice_fraction_values = [], []
    count = 0
    for fp in paths:
        match = re.search(r"checkpoint_(\d+)\.npz", fp)
        if not match:
            continue
        n = int(match.group(1))
        if max_checkpoints is not None and count >= max_checkpoints:
            break
        u = np.load(fp)["u"]
        ice_fraction_values.append(np.mean(u < cfg.u_pt_nd))
        times.append(n * cfg.geometry.dt)
        count += 1
    return np.array(times), np.array(ice_fraction_values)


t_cold_nu, nu_cold = load_nusselt_data(
    "data/cold_start_full/checkpoint_*.npz", max_checkpoints=39
)
t_warm_nu, nu_warm = load_nusselt_data(
    "data/warm_start_full/checkpoint_*.npz", max_checkpoints=39
)

t_cold_ice, ice_fraction_cold = load_ice_fraction_data(
    "data/cold_start_full/checkpoint_*.npz"
)
t_warm_ice, ice_fraction_warm = load_ice_fraction_data(
    "data/warm_start_full/checkpoint_*.npz"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.15))

ax1.plot(t_cold_nu, nu_cold, linewidth=1.5, color="tab:blue")
ax1.plot(t_warm_nu, nu_warm, linewidth=1.5, color="tab:orange")
ax1.set_xlabel(r"Time, s")
ax1.set_ylabel(r"Nu")

L = 0.5
i1 = int(0.2 * (len(t_cold_nu) - 1))
x1, y1 = t_cold_nu[i1], nu_cold[i1]
ax1.plot([x1, x1], [y1, y1 + L], color="black", linewidth=0.8)
ax1.text(x1 - 30.0, y1 + L + 0.2, "1", va="center", fontsize=10)

L = 0.5
i1 = int(0.2 * (len(t_cold_nu) - 1))
x1, y1 = t_cold_nu[i1], nu_warm[i1]
ax1.plot([x1, x1 + 100], [y1, y1 - 0.4], color="black", linewidth=0.8)
ax1.text(x1 + 100, y1 - 0.6, "2", va="center", fontsize=10)

add_subfigure_label(ax1, "a")

ax2.plot(t_cold_ice, ice_fraction_cold, linewidth=1.5, color="tab:blue")
ax2.plot(t_warm_ice, ice_fraction_warm, linewidth=1.5, color="tab:orange")
ax2.set_xlabel(r"Time, s")
ax2.set_ylabel(r"Ice fraction")

i1 = int(0.15 * (len(t_cold_ice) - 1))
x1, y1 = t_cold_ice[i1], ice_fraction_cold[i1]
ax2.plot([x1, x1 - 50.0], [y1, y1 + 0.04], color="black", linewidth=0.8)
ax2.text(x1 - 130.0, y1 + 0.05, "1", va="center", fontsize=10)

i1 = int(0.2 * (len(t_cold_ice) - 1))
x1, y1 = t_cold_ice[i1], ice_fraction_warm[i1]
ax2.plot([x1, x1 + 100], [y1, y1 - 0.04], color="black", linewidth=0.8)
ax2.text(x1 + 50, y1 - 0.055, "2", va="center", fontsize=10)

add_subfigure_label(ax2, "b")

plt.tight_layout()
plt.savefig("./graphs/nu_ice_f_combined_v2.tiff", dpi=300, bbox_inches="tight")
plt.show()

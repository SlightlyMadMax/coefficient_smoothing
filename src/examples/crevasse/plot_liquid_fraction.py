import matplotlib as mpl
import glob
import re

import numpy as np
from matplotlib import pyplot as plt

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

cfg_conv: ExperimentConfig = ExperimentConfig.load_from_file("./convection/config.json")
cfg_cond: ExperimentConfig = ExperimentConfig.load_from_file("./conduction/config.json")

s_f_conv = []
s_f_cond = []

times_conv = []
times_cond = []

conv_mask = "./data/convection/48_hrs/checkpoint_*.npz"
conv_paths = sorted(
    glob.glob(conv_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cond_mask = "./data/conduction/checkpoint_*.npz"
cond_paths = sorted(
    glob.glob(cond_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

for file_path in conv_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    s_f_conv.append(np.mean(u < cfg_conv.u_pt_nd))
    times_conv.append(n * cfg_conv.geometry.dt)

for file_path in cond_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    s_f_cond.append(np.mean(u < cfg_cond.u_pt_nd))
    times_cond.append(n * cfg_cond.geometry.dt)

fig, ax = plt.subplots(figsize=(6.3, 3.9))

(line_conv,) = ax.plot(times_conv, s_f_conv)
(line_cond,) = ax.plot(times_cond, s_f_cond)

ax.set_xlabel(r"Time, s")
ax.set_ylabel(r"Ice fraction")

idx_conv = len(times_conv) // 2
idx_cond = len(times_cond) // 2

x_conv, y_conv = times_conv[idx_conv], s_f_conv[idx_conv]
x_cond, y_cond = times_cond[idx_cond], s_f_cond[idx_cond]

dx_conv = (max(times_conv) - min(times_conv)) * 0.03
dx_cond = (max(times_cond) - min(times_cond)) * 0.03
dy = 0.025
text_offset = 0.015

ax.plot([x_conv, x_conv - dx_conv], [y_conv, y_conv - dy], color="black", linewidth=0.8)
ax.text(
    x_conv - dx_conv - 2000,
    y_conv - dy - text_offset,
    "2",
    ha="center",
    va="top",
    fontsize=12,
    family="serif",
)

ax.plot([x_cond, x_cond - dx_cond], [y_cond, y_cond - dy], color="black", linewidth=0.8)
ax.text(
    x_cond - dx_cond - 2000,
    y_cond - dy - text_offset,
    "1",
    ha="center",
    va="top",
    fontsize=12,
    family="serif",
)

plt.tight_layout()
fig.savefig("./graphs/ice_fraction.png", dpi=300, bbox_inches="tight")
plt.show()

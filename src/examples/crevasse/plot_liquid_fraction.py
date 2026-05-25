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

plt.figure(figsize=(6.3, 3.9))
plt.plot(times_conv, s_f_conv, label=r"Convection")
plt.plot(times_cond, s_f_cond, label=r"Pure conduction")

plt.xlabel(r"Time, s")
plt.ylabel(r"Ice fraction")
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/ice_fraction.png")
plt.show()
